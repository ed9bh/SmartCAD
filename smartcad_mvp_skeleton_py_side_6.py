#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMARTCAD — MVP Skeleton (v0.2c)
Stack: Python 3.10+, PySide6

Novidades v0.2c:
- **Criação por clique no viewport**
  - Ponto 3D (clicar): um clique no plano Z=0 cria o ponto.
  - Polyline 3D (clicar): múltiplos cliques para os vértices; **Enter**/**Duplo-clique**/**Botão direito** finaliza; **Backspace** desfaz último; **Esc** cancela.
  - Pré-visualização na tela enquanto desenha.
  - **Snap opcional à grade** (Exibição → Ativar Snap à Grade).
- **Polyline 3D (Planilha)**
  - Dialogo com **tabela (X,Y,Z)**, botões **Adicionar/Remover linha**, total de **comprimento**.
  - Aceita colar de Excel/CSV (padrão do QTableWidget).
- Ajustes de OpenGL mantidos (imports corretos + contexto 3.3 Core + constantes GL).

Como rodar:
  pip install PySide6 numpy
  python smartcad_main.py
"""
from __future__ import annotations
import json
import sys
import time
import math
import numpy as np
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any, Callable

from PySide6 import QtCore, QtGui, QtWidgets
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QMatrix4x4, QSurfaceFormat
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtOpenGL import (
    QOpenGLShaderProgram, QOpenGLShader, QOpenGLVertexArrayObject, QOpenGLBuffer
)
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QFileDialog, QMessageBox, QDockWidget,
    QListWidget, QWidget, QVBoxLayout, QLabel, QFormLayout, QLineEdit,
    QPushButton, QHBoxLayout, QDoubleSpinBox, QMenu, QInputDialog,
    QTableWidget, QTableWidgetItem, QHeaderView, QCheckBox
)

# ==========================
# Constantes OpenGL (sem PyOpenGL)
# ==========================
GL_COLOR_BUFFER_BIT = 0x00004000
GL_DEPTH_BUFFER_BIT = 0x00000100
GL_POINTS = 0x0000
GL_LINES = 0x0001
GL_DEPTH_TEST = 0x0B71
GL_FLOAT = 0x1406

# ==========================
# Utilidades
# ==========================
APP_SCHEMA_VERSION = "smartcad.project@1"
ENTITY_SCHEMA_VERSION = "smartcad.entity@1"

# ==========================
# Núcleo: Journal (simples)
# ==========================
@dataclass
class JournalOp:
    op: str
    target: str
    before: Dict[str, Any] | None
    after: Dict[str, Any] | None
    ts: float = field(default_factory=lambda: time.time())

class Journal:
    def __init__(self, project_root: Path):
        self.root = project_root
        self.log_path = self.root/".smartcad"/"journal.log"
        self.stack: List[JournalOp] = []
        self.redo_stack: List[JournalOp] = []
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, op: JournalOp):
        self.stack.append(op)
        self.redo_stack.clear()
        with self.log_path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(asdict(op), ensure_ascii=False) + "\n")

    def undo(self) -> Optional[JournalOp]:
        if not self.stack:
            return None
        op = self.stack.pop()
        self.redo_stack.append(op)
        return op

    def redo(self) -> Optional[JournalOp]:
        if not self.redo_stack:
            return None
        op = self.redo_stack.pop()
        self.stack.append(op)
        return op

# ==========================
# Núcleo: Projeto & Camadas
# ==========================
class ProjectManager:
    def __init__(self):
        self.root: Optional[Path] = None
        self.data: Dict[str, Any] = {}
        self.journal: Optional[Journal] = None
        self.layer_manager: Optional[LayerManager] = None

    def has_project(self) -> bool:
        return self.root is not None

    def create_structure(self, root: Path, name: str, epsg: Optional[int] = None):
        """Cria estrutura de diretórios do projeto."""
        self.root = root
        (root/".smartcad"/"versions").mkdir(parents=True, exist_ok=True)
        (root/".smartcad"/"locks").mkdir(parents=True, exist_ok=True)
        (root/".smartcad"/"resources").mkdir(parents=True, exist_ok=True)
        (root/"layers"/"_default").mkdir(parents=True, exist_ok=True)
        (root/"imports").mkdir(parents=True, exist_ok=True)
        (root/"exports").mkdir(parents=True, exist_ok=True)
        (root/"cache").mkdir(parents=True, exist_ok=True)

        self.data = {
            "schema": APP_SCHEMA_VERSION,
            "name": name,
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "units": {"linear": "meters", "angular": "degrees"},
            # EPSG opcional para UTM no Brasil
            "epsg": epsg,  # pode ser None
            "activeLayer": "_default",
            "plugins": ["topografia", "projetos"],
            "bounds": {"min": [-1000,-1000,-100], "max": [1000,1000,200]},
        }
        with (root/".smartcad"/"project.json").open('w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        self.journal = Journal(root)
        self.layer_manager = LayerManager(root)

    def open_project(self, root: Path):
        p = root/".smartcad"/"project.json"
        if not p.exists():
            raise FileNotFoundError("project.json não encontrado no diretório selecionado")
        self.root = root
        with p.open('r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.journal = Journal(root)
        self.layer_manager = LayerManager(root)

    def save(self):
        if not self.root:
            return
        p = self.root/".smartcad"/"project.json"
        with p.open('w', encoding='utf-8') as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)

class LayerManager:
    def __init__(self, project_root: Path):
        self.root = Path(project_root)/"layers"
        self.active = "_default"

    def list_layers(self) -> List[str]:
        if not self.root.exists():
            return []
        return [p.name for p in self.root.iterdir() if p.is_dir()]

    def create(self, name: str):
        safe = sanitize_name(name)
        (self.root/safe).mkdir(parents=True, exist_ok=True)

    def remove(self, name: str):
        p = self.root/name
        if p.exists() and p.is_dir():
            # Apenas remove se vazio para evitar perda acidental
            try:
                next(p.iterdir())
                raise OSError("Camada não está vazia")
            except StopIteration:
                p.rmdir()

    def move_entity_file(self, eid: str, src_layer: str, dst_layer: str):
        src = self.root/src_layer/f"{eid}.entity.json"
        dst = self.root/dst_layer/f"{eid}.entity.json"
        if not src.exists():
            raise FileNotFoundError(f"Entidade {eid} não encontrada em {src_layer}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        src.rename(dst)

# ==========================
# Núcleo: Entidades (dados)
# ==========================

def sanitize_name(name: str) -> str:
    return "".join(c for c in name if c.isalnum() or c in ("_","-"," ")).strip().replace(" ", "_")

@dataclass
class Transform:
    t: Tuple[float,float,float] = (0.0, 0.0, 0.0)
    r: Tuple[float,float,float] = (0.0, 0.0, 0.0)
    s: Tuple[float,float,float] = (1.0, 1.0, 1.0)

@dataclass
class Style:
    color: str = "#ffffff"
    linetype: str = "continuous"

    @staticmethod
    def default():
        return Style()

@dataclass
class Entity:
    id: str
    type: str
    layer: str
    name: str = ""
    transform: Transform = field(default_factory=Transform)
    style: Style = field(default_factory=Style)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> Dict[str, Any]:
        return {
            "schema": ENTITY_SCHEMA_VERSION,
            "id": self.id,
            "type": self.type,
            "layer": self.layer,
            "name": self.name,
            "transform": asdict(self.transform),
            "style": asdict(self.style),
            "metadata": self.metadata,
        }

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'Entity':
        base = Entity(
            id=data["id"], type=data["type"], layer=data["layer"], name=data.get("name",""),
            transform=Transform(**data.get("transform", {})),
            style=Style(**data.get("style", {})),
            metadata=data.get("metadata", {})
        )
        return base

@dataclass
class Point3D(Entity):
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def to_json(self) -> Dict[str, Any]:
        d = super().to_json()
        d.update({"data": {"x": self.x, "y": self.y, "z": self.z}})
        return d

@dataclass
class Polyline3D(Entity):
    vertices: List[Tuple[float,float,float]] = field(default_factory=list)
    closed: bool = False

    def to_json(self) -> Dict[str, Any]:
        d = super().to_json()
        d.update({"data": {"vertices": self.vertices, "closed": self.closed}})
        return d

# ==========================
# Persistência de Entidades
# ==========================
class EntityStore:
    def __init__(self, project_root: Path):
        self.root = Path(project_root)/"layers"
        self.counter_path = Path(project_root)/".smartcad"/"id_counter.txt"
        self.counter_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.counter_path.exists():
            self.counter_path.write_text("0", encoding='utf-8')

    def new_id(self) -> str:
        n = int(self.counter_path.read_text(encoding='utf-8').strip() or 0)
        n += 1
        self.counter_path.write_text(str(n), encoding='utf-8')
        return f"e-{n:06d}"

    def save(self, ent: Entity):
        layer_dir = self.root/ent.layer
        layer_dir.mkdir(parents=True, exist_ok=True)
        path = layer_dir/f"{ent.id}.entity.json"
        tmp = path.with_suffix('.tmp')
        tmp.write_text(json.dumps(ent.to_json(), ensure_ascii=False, indent=2), encoding='utf-8')
        tmp.replace(path)  # atômico na maioria dos FS

    def delete(self, ent: Entity):
        path = self.root/ent.layer/f"{ent.id}.entity.json"
        if path.exists():
            path.unlink()

    def load_all(self) -> List[Entity]:
        ents: List[Entity] = []
        for layer in self.root.glob("*/"):
            for f in layer.glob("*.entity.json"):
                try:
                    data = json.loads(f.read_text(encoding='utf-8'))
                    ent = Entity.from_json(data)
                    # Carregar dados específicos
                    if ent.type == 'Point3D':
                        d = data.get('data',{})
                        ent = Point3D(**asdict(ent), x=d.get('x',0.0), y=d.get('y',0.0), z=d.get('z',0.0))
                    elif ent.type == 'Polyline3D':
                        d = data.get('data',{})
                        ent = Polyline3D(**asdict(ent), vertices=d.get('vertices',[]), closed=d.get('closed',False))
                    ents.append(ent)
                except Exception:
                    continue
        return ents

# ==========================
# Viewport 3D (OpenGL básico + desenho interativo)
# ==========================
class Viewport3D(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(900, 600)
        self.setFocusPolicy(Qt.StrongFocus)
        self.camera_dist = 50.0
        self.rot_x = -30.0
        self.rot_y = 45.0
        self.last_pos: Optional[QtCore.QPoint] = None
        self.entities: List[Entity] = []
        self.grid_size = 100
        self.grid_step = 5

        # Interação de desenho
        self.mode: str = 'idle'  # 'idle'|'point'|'pline'
        self.snap_enabled: bool = True
        self.temp_vertices: List[Tuple[float,float,float]] = []
        self._proj_np = None
        self._view_np = None
        self._mvp_np = None
        self._eye = np.array([0,0,50.0], dtype=float)
        self._on_point_created: Optional[Callable[[Tuple[float,float,float]], None]] = None
        self._on_polyline_created: Optional[Callable[[List[Tuple[float,float,float]]], None]] = None

    def set_entities(self, ents: List[Entity]):
        self.entities = ents
        self.update()

    # ---------- Modo desenho ----------
    def start_point_mode(self, on_created: Callable[[Tuple[float,float,float]], None]):
        self.mode = 'point'
        self._on_point_created = on_created
        self.temp_vertices.clear()
        self.setCursor(Qt.CrossCursor)
        self.setFocus()

    def start_polyline_mode(self, on_created: Callable[[List[Tuple[float,float,float]]], None]):
        self.mode = 'pline'
        self._on_polyline_created = on_created
        self.temp_vertices.clear()
        self.setCursor(Qt.CrossCursor)
        self.setFocus()

    def finish_polyline(self):
        if self.mode == 'pline' and len(self.temp_vertices) >= 2 and self._on_polyline_created:
            verts = list(self.temp_vertices)
            self._on_polyline_created(verts)
        self.cancel_drawing()

    def cancel_drawing(self):
        self.mode = 'idle'
        self._on_point_created = None
        self._on_polyline_created = None
        self.temp_vertices.clear()
        self.unsetCursor()
        self.update()

    # ---------- Eventos mouse/teclado ----------
    def mousePressEvent(self, e: QtGui.QMouseEvent):
        if self.mode in ('point','pline') and e.button() == Qt.LeftButton:
            p = self._pick_plane_point(e.position().x(), e.position().y())
            if p is not None:
                if self.snap_enabled:
                    p = self._snap_point(p)
                if self.mode == 'point' and self._on_point_created:
                    self._on_point_created(p)
                    self.cancel_drawing()
                    return
                elif self.mode == 'pline':
                    self.temp_vertices.append(p)
                    self.update()
                    return
        if self.mode == 'pline' and e.button() == Qt.RightButton:
            if len(self.temp_vertices) >= 2:
                self.finish_polyline()
                return
            else:
                self.cancel_drawing()
                return
        # Rotação da câmera
        self.last_pos = e.position().toPoint()

    def mouseDoubleClickEvent(self, e: QtGui.QMouseEvent):
        if self.mode == 'pline' and e.button() == Qt.LeftButton:
            if len(self.temp_vertices) >= 2:
                self.finish_polyline()
                return

    def mouseMoveEvent(self, e: QtGui.QMouseEvent):
        if self.mode not in ('point','pline'):
            if self.last_pos is None:
                return
            dx = e.position().x() - self.last_pos.x()
            dy = e.position().y() - self.last_pos.y()
            if e.buttons() & Qt.LeftButton:
                self.rot_y += dx * 0.4
                self.rot_x += dy * 0.4
                self.update()
            self.last_pos = e.position().toPoint()
        else:
            # Atualiza preview do último ponto (não permanente)
            self.update()

    def wheelEvent(self, e: QtGui.QWheelEvent):
        delta = e.angleDelta().y() / 120.0
        self.camera_dist *= (0.9 ** delta)
        self.camera_dist = max(5.0, min(1000.0, self.camera_dist))
        self.update()

    def keyPressEvent(self, e: QtGui.QKeyEvent):
        if self.mode == 'pline':
            if e.key() in (Qt.Key_Return, Qt.Key_Enter):
                if len(self.temp_vertices) >= 2:
                    self.finish_polyline()
                    return
            elif e.key() == Qt.Key_Escape:
                self.cancel_drawing()
                return
            elif e.key() == Qt.Key_Backspace:
                if self.temp_vertices:
                    self.temp_vertices.pop()
                    self.update()
                    return
        elif self.mode == 'point':
            if e.key() == Qt.Key_Escape:
                self.cancel_drawing()
                return
        super().keyPressEvent(e)

    # ---------- OpenGL ----------
    def initializeGL(self):
        gl = self.context().functions()
        gl.glEnable(GL_DEPTH_TEST)
        gl.glClearColor(0.1, 0.1, 0.12, 1.0)

    def resizeGL(self, w: int, h: int):
        gl = self.context().functions()
        gl.glViewport(0, 0, w, max(1,h))

    def paintGL(self):
        gl = self.context().functions()
        gl.glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # Matriz de projeção simples (perspectiva)
        w = max(1, self.width())
        h = max(1, self.height())
        aspect = w / h
        proj = perspective(math.radians(60.0), aspect, 0.1, 5000.0)

        # Matriz de view (câmera) — órbita
        eye = vec3_from_orbit(self.camera_dist, math.radians(self.rot_x), math.radians(self.rot_y))
        view = look_at(eye, (0,0,0), (0,0,1))

        mvp = mat4_mul(proj, view)  # sem model matrix (identidade)

        # Guardar versões numpy para unproject
        self._proj_np = np.array(proj, dtype=float)
        self._view_np = np.array(view, dtype=float)
        self._mvp_np = self._proj_np @ self._view_np
        self._eye = np.array(eye, dtype=float)

        # Grid
        self.draw_grid(gl, mvp)
        # Entidades
        self.draw_entities(gl, mvp)
        # Preview de desenho
        self.draw_preview(gl, mvp)

    def draw_grid(self, gl, mvp):
        lines = []
        s = self.grid_size
        step = self.grid_step
        for i in range(-s, s+1, step):
            lines.append(((i, -s, 0), (i, s, 0)))
            lines.append(((-s, i, 0), (s, i, 0)))
        draw_lines(gl, lines, mvp)

    def draw_entities(self, gl, mvp):
        pts = []
        polylines: List[List[Tuple[float,float,float]]] = []
        for e in self.entities:
            if isinstance(e, Point3D):
                pts.append((e.x, e.y, e.z))
            elif isinstance(e, Polyline3D):
                if len(e.vertices) >= 2:
                    polylines.append(e.vertices)
        draw_points(gl, pts, mvp, size=6)
        for pl in polylines:
            segs = list(zip(pl[:-1], pl[1:]))
            draw_lines(gl, segs, mvp)

    def draw_preview(self, gl, mvp):
        if self.mode == 'pline' and self.temp_vertices:
            segs = list(zip(self.temp_vertices[:-1], self.temp_vertices[1:]))
            draw_lines(gl, segs, mvp)
            draw_points(gl, [self.temp_vertices[-1]], mvp, size=7)
        elif self.mode == 'point' and self.temp_vertices:
            draw_points(gl, [self.temp_vertices[-1]], mvp, size=7)

    # ---------- Picking util ----------
    def _pick_plane_point(self, sx: float, sy: float, plane_z: float = 0.0) -> Optional[Tuple[float,float,float]]:
        if self._mvp_np is None:
            return None
        w = max(1, self.width())
        h = max(1, self.height())
        # NDC
        x_ndc = (2.0 * sx / w) - 1.0
        y_ndc = 1.0 - (2.0 * sy / h)
        near_clip = np.array([x_ndc, y_ndc, -1.0, 1.0])
        far_clip  = np.array([x_ndc, y_ndc,  1.0, 1.0])
        inv = np.linalg.inv(self._mvp_np)
        near_world = inv @ near_clip
        far_world  = inv @ far_clip
        if abs(near_world[3]) < 1e-6 or abs(far_world[3]) < 1e-6:
            return None
        near_world = near_world[:3] / near_world[3]
        far_world  = far_world[:3] / far_world[3]
        ray_dir = far_world - near_world
        ray_dir = ray_dir / (np.linalg.norm(ray_dir) + 1e-12)
        # Interseção com plano z = plane_z
        if abs(ray_dir[2]) < 1e-9:
            return None
        t = (plane_z - near_world[2]) / ray_dir[2]
        if t < 0:
            return None
        p = near_world + t * ray_dir
        self.temp_vertices_preview = (float(p[0]), float(p[1]), float(p[2]))
        return (float(p[0]), float(p[1]), float(p[2]))

    def _snap_point(self, p: Tuple[float,float,float]) -> Tuple[float,float,float]:
        step = float(self.grid_step)
        return (
            round(p[0] / step) * step,
            round(p[1] / step) * step,
            round(p[2] / step) * step,
        )

# ==========================
# Render helpers (shader pipeline mínimo)
# ==========================
import struct

def draw_points(gl, points: List[Tuple[float,float,float]], mvp, size=5):
    if not points:
        return
    gl.glPointSize(size)
    draw_primitives(gl, GL_POINTS, points, mvp)


def draw_lines(gl, segments: List[Tuple[Tuple[float,float,float], Tuple[float,float,float]]], mvp):
    if not segments:
        return
    verts: List[Tuple[float,float,float]] = []
    for a,b in segments:
        verts.append(a)
        verts.append(b)
    draw_primitives(gl, GL_LINES, verts, mvp)


def draw_primitives(gl, mode, vertices: List[Tuple[float,float,float]], mvp):
    # Programa de shader mínimo (uma única instância global)
    if not hasattr(draw_primitives, "prog"):
        prog = QOpenGLShaderProgram()
        vs = QOpenGLShader(QOpenGLShader.Vertex)
        fs = QOpenGLShader(QOpenGLShader.Fragment)
        vs_src = """
        #version 330 core
        layout(location=0) in vec3 in_pos;
        uniform mat4 u_mvp;
        void main(){
            gl_Position = u_mvp * vec4(in_pos,1.0);
        }
        """
        fs_src = """
        #version 330 core
        out vec4 FragColor;
        void main(){ FragColor = vec4(0.95,0.95,0.98,1.0); }
        """
        vs.compileSourceCode(vs_src)
        fs.compileSourceCode(fs_src)
        prog.addShader(vs)
        prog.addShader(fs)
        prog.link()
        draw_primitives.prog = prog
        draw_primitives.vao = None
        draw_primitives.vbo = None
    prog: QOpenGLShaderProgram = draw_primitives.prog
    if draw_primitives.vao is None:
        draw_primitives.vao = QOpenGLVertexArrayObject()
        draw_primitives.vao.create()
        draw_primitives.vbo = QOpenGLBuffer(QOpenGLBuffer.VertexBuffer)
        draw_primitives.vbo.create()

    vao: QOpenGLVertexArrayObject = draw_primitives.vao
    vbo: QOpenGLBuffer = draw_primitives.vbo

    vao.bind()
    vbo.bind()

    data = struct.pack(f"{len(vertices)*3}f", *sum(([x,y,z] for x,y,z in vertices), []))
    vbo.allocate(data, len(data))

    prog.bind()
    loc = 0
    prog.enableAttributeArray(loc)
    prog.setAttributeBuffer(loc, GL_FLOAT, 0, 3)

    # Envia matriz MVP (column-major)
    mvp_flat = [c for col in mvp for c in col]
    prog.setUniformValue("u_mvp", QMatrix4x4(*mvp_flat))

    gl.glDrawArrays(mode, 0, len(vertices))

    prog.disableAttributeArray(loc)
    prog.release()

    vbo.release()
    vao.release()

# ==========================
# Matemática 3D simples
# ==========================

def perspective(fovy, aspect, znear, zfar):
    f = 1.0 / math.tan(fovy/2.0)
    m = [
        [f/aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (zfar+znear)/(znear-zfar), (2*zfar*znear)/(znear-zfar)],
        [0, 0, -1, 0]
    ]
    return m


def mat4_mul(a, b):
    r = [[0]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            r[i][j] = sum(a[i][k]*b[k][j] for k in range(4))
    return r


def look_at(eye, target, up):
    eye = np.array(eye, dtype=float)
    target = np.array(target, dtype=float)
    up = np.array(up, dtype=float)
    f = target - eye
    f = f/np.linalg.norm(f)
    s = np.cross(f, up)
    s = s/np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    t = np.identity(4)
    t[0, 3] = -eye[0]
    t[1, 3] = -eye[1]
    t[2, 3] = -eye[2]
    res = m @ t
    return [[res[i,j] for j in range(4)] for i in range(4)]


def vec3_from_orbit(dist, pitch, yaw):
    x = dist * math.cos(pitch) * math.cos(yaw)
    y = dist * math.cos(pitch) * math.sin(yaw)
    z = dist * math.sin(pitch)
    return (x, y, z)

# ==========================
# Painéis (Docks)
# ==========================
class LayersDock(QDockWidget):
    def __init__(self, pm: ProjectManager, on_layer_change, parent=None):
        super().__init__("Camadas", parent)
        self.pm = pm
        self.on_layer_change = on_layer_change
        self.listw = QListWidget()
        self.setWidget(self.listw)
        self.listw.itemSelectionChanged.connect(self._sel_changed)
        self.setAllowedAreas(Qt.LeftDockWidgetArea)
        self.refresh()

    def refresh(self):
        self.listw.clear()
        if self.pm.layer_manager:
            for name in sorted(self.pm.layer_manager.list_layers()):
                self.listw.addItem(name)
            for i in range(self.listw.count()):
                if self.listw.item(i).text() == (self.pm.layer_manager.active if self.pm.layer_manager else "_default"):
                    self.listw.setCurrentRow(i)
                    break

    def _sel_changed(self):
        it = self.listw.currentItem()
        if it:
            name = it.text()
            if self.pm.layer_manager:
                self.pm.layer_manager.active = name
                self.on_layer_change(name)

# ==========================
# Dialogs de criação de entidades (formulários)
# ==========================
class NewPointDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Novo Ponto 3D")
        form = QFormLayout(self)
        self.x = QDoubleSpinBox(); self.x.setRange(-1e9, 1e9)
        self.y = QDoubleSpinBox(); self.y.setRange(-1e9, 1e9)
        self.z = QDoubleSpinBox(); self.z.setRange(-1e9, 1e9)
        form.addRow("X:", self.x)
        form.addRow("Y:", self.y)
        form.addRow("Z:", self.z)
        btns = QHBoxLayout()
        ok = QPushButton("Criar"); cancel = QPushButton("Cancelar")
        btns.addWidget(ok); btns.addWidget(cancel)
        form.addRow(btns)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

class NewPolylineDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Nova Polyline 3D")
        vbox = QVBoxLayout(self)
        self.edit = QLineEdit()
        self.edit.setPlaceholderText("Vértices: x1,y1,z1; x2,y2,z2; ...")
        vbox.addWidget(QLabel("Informe os vértices (separados por ponto e vírgula):"))
        vbox.addWidget(self.edit)
        btns = QHBoxLayout()
        ok = QPushButton("Criar"); cancel = QPushButton("Cancelar")
        btns.addWidget(ok); btns.addWidget(cancel)
        vbox.addLayout(btns)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)

    def vertices(self) -> List[Tuple[float,float,float]]:
        txt = self.edit.text().strip()
        verts: List[Tuple[float,float,float]] = []
        if not txt:
            return verts
        for part in txt.split(';'):
            if not part.strip():
                continue
            nums = [float(n) for n in part.split(',')]
            if len(nums) == 3:
                verts.append((nums[0], nums[1], nums[2]))
        return verts

class PolylineTableDialog(QtWidgets.QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Polyline 3D — Planilha")
        self.resize(520, 420)
        vbox = QVBoxLayout(self)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["X","Y","Z"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        vbox.addWidget(self.table)

        btns1 = QHBoxLayout()
        self.btn_add = QPushButton("Adicionar linha")
        self.btn_del = QPushButton("Remover linha")
        btns1.addWidget(self.btn_add)
        btns1.addWidget(self.btn_del)
        vbox.addLayout(btns1)

        self.lbl_len = QLabel("Comprimento: 0.000 m")
        vbox.addWidget(self.lbl_len)

        btns2 = QHBoxLayout()
        ok = QPushButton("Criar")
        cancel = QPushButton("Cancelar")
        btns2.addWidget(ok); btns2.addWidget(cancel)
        vbox.addLayout(btns2)

        self.btn_add.clicked.connect(self.add_row)
        self.btn_del.clicked.connect(self.del_row)
        ok.clicked.connect(self.accept)
        cancel.clicked.connect(self.reject)
        self.table.itemChanged.connect(self.update_length)

        # duas linhas iniciais
        self.add_row(); self.add_row()

    def add_row(self):
        r = self.table.rowCount()
        self.table.insertRow(r)
        for c in range(3):
            self.table.setItem(r, c, QTableWidgetItem("0"))
        self.update_length()

    def del_row(self):
        r = self.table.currentRow()
        if r >= 0:
            self.table.removeRow(r)
            self.update_length()

    def vertices(self) -> List[Tuple[float,float,float]]:
        verts: List[Tuple[float,float,float]] = []
        for r in range(self.table.rowCount()):
            try:
                x = float(self.table.item(r,0).text()) if self.table.item(r,0) else 0.0
                y = float(self.table.item(r,1).text()) if self.table.item(r,1) else 0.0
                z = float(self.table.item(r,2).text()) if self.table.item(r,2) else 0.0
                verts.append((x,y,z))
            except Exception:
                pass
        return verts

    def update_length(self):
        vs = self.vertices()
        if len(vs) < 2:
            self.lbl_len.setText("Comprimento: 0.000 m")
            return
        length = 0.0
        for a,b in zip(vs[:-1], vs[1:]):
            ax,ay,az = a; bx,by,bz = b
            length += math.dist((ax,ay,az),(bx,by,bz))
        self.lbl_len.setText(f"Comprimento: {length:.3f} m")

# ==========================
# Janela Principal
# ==========================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SMARTCAD — MVP 0.2c")
        self.resize(1400, 860)

        self.pm = ProjectManager()
        self.entity_store: Optional[EntityStore] = None

        # Viewport central
        self.viewport = Viewport3D(self)
        self.setCentralWidget(self.viewport)

        # Docks
        self.layers_dock = LayersDock(self.pm, self.on_layer_changed, self)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.layers_dock)

        # Menus
        self._build_menus()
        self._update_actions_enabled(False)

    # -------- Menus --------
    def _build_menus(self):
        mb = self.menuBar()

        # Projeto
        m_proj = mb.addMenu("Projeto")
        act_new = QAction("Novo Projeto", self, triggered=self.action_new_project)
        act_open = QAction("Abrir Projeto", self, triggered=self.action_open_project)
        act_save = QAction("Salvar", self, triggered=self.action_save_project)
        m_proj.addAction(act_new)
        m_proj.addAction(act_open)
        m_proj.addSeparator()
        m_proj.addAction(act_save)

        # Camadas
        m_layers = mb.addMenu("Camadas")
        self.act_new_layer = QAction("Nova Camada", self, triggered=self.action_new_layer)
        self.act_del_layer = QAction("Remover Camada (se vazia)", self, triggered=self.action_del_layer)
        m_layers.addAction(self.act_new_layer)
        m_layers.addAction(self.act_del_layer)

        # Inserir
        m_ins = mb.addMenu("Inserir")
        self.act_new_point = QAction("Ponto 3D (formulário)", self, triggered=self.action_new_point)
        self.act_new_point_click = QAction("Ponto 3D (clicar)", self, triggered=self.action_new_point_click)
        self.act_new_pline = QAction("Polyline 3D (formulário)", self, triggered=self.action_new_polyline)
        self.act_new_pline_click = QAction("Polyline 3D (clicar)", self, triggered=self.action_new_polyline_click)
        self.act_new_pline_table = QAction("Polyline 3D (planilha)", self, triggered=self.action_new_polyline_table)
        m_ins.addAction(self.act_new_point)
        m_ins.addAction(self.act_new_point_click)
        m_ins.addSeparator()
        m_ins.addAction(self.act_new_pline)
        m_ins.addAction(self.act_new_pline_click)
        m_ins.addAction(self.act_new_pline_table)

        # Topografia/Projetos (placeholders)
        mb.addMenu("Topografia")
        mb.addMenu("Projetos")

        # Exibição
        m_view = mb.addMenu("Exibição")
        self.act_reset_view = QAction("Resetar Visão", self, triggered=self.action_reset_view)
        self.act_snap = QAction("Ativar Snap à Grade", self, checkable=True, checked=True)
        self.act_snap.toggled.connect(lambda on: self.toggle_snap(on))
        m_view.addAction(self.act_reset_view)
        m_view.addAction(self.act_snap)

        # Import/Export (placeholders)
        mb.addMenu("Import/Export")

        # Ajuda
        m_help = mb.addMenu("Ajuda")
        m_help.addAction("Sobre", lambda: QMessageBox.information(self, "Sobre", "SMARTCAD MVP 0.2c\nEditor 3D baseado em diretório"))

        self.actions_project = [act_save, self.act_new_layer, self.act_del_layer,
                                self.act_new_point, self.act_new_point_click,
                                self.act_new_pline, self.act_new_pline_click,
                                self.act_new_pline_table, self.act_reset_view, self.act_snap]

    def toggle_snap(self, on: bool):
        self.viewport.snap_enabled = on

    def _update_actions_enabled(self, enabled: bool):
        for a in getattr(self, 'actions_project', []):
            a.setEnabled(enabled)

    # -------- Ações de Projeto --------
    def action_new_project(self):
        root = QFileDialog.getExistingDirectory(self, "Escolha o diretório do novo projeto")
        if not root:
            return
        name, ok = QInputDialog.getText(self, "Nome do Projeto", "Nome:")
        if not ok or not name:
            return
        epsg_text, ok = QInputDialog.getText(self, "EPSG (opcional)", "Informe EPSG UTM (ex.: 31983) ou deixe vazio:")
        epsg = int(epsg_text) if ok and epsg_text.strip().isdigit() else None
        self.pm.create_structure(Path(root), name, epsg)
        self.entity_store = EntityStore(Path(root))
        self.layers_dock.refresh()
        self.reload_entities()
        self._update_actions_enabled(True)

    def action_open_project(self):
        root = QFileDialog.getExistingDirectory(self, "Abrir diretório do projeto")
        if not root:
            return
        try:
            self.pm.open_project(Path(root))
        except Exception as e:
            QMessageBox.critical(self, "Erro", str(e))
            return
        self.entity_store = EntityStore(Path(root))
        self.layers_dock.refresh()
        self.reload_entities()
        self._update_actions_enabled(True)

    def action_save_project(self):
        self.pm.save()
        QMessageBox.information(self, "Salvar", "Projeto salvo com sucesso.")

    # -------- Ações de Camadas --------
    def action_new_layer(self):
        if not self.pm.layer_manager:
            return
        name, ok = QInputDialog.getText(self, "Nova Camada", "Nome:")
        if not ok or not name:
            return
        self.pm.layer_manager.create(name)
        self.layers_dock.refresh()
        self.reload_entities()

    def action_del_layer(self):
        if not self.pm.layer_manager:
            return
        cur = self.pm.layer_manager.active
        if cur == "_default":
            QMessageBox.warning(self, "Camadas", "Camada _default não pode ser removida.")
            return
        try:
            self.pm.layer_manager.remove(cur)
            self.pm.layer_manager.active = "_default"
            self.layers_dock.refresh()
            self.reload_entities()
        except OSError as e:
            QMessageBox.warning(self, "Camadas", str(e))

    def on_layer_changed(self, name: str):
        self.statusBar().showMessage(f"Camada ativa: {name}")

    # -------- Ações de Inserção --------
    def action_new_point(self):
        if not (self.pm.layer_manager and self.entity_store):
            return
        dlg = NewPointDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            eid = self.entity_store.new_id()
            ent = Point3D(id=eid, type='Point3D', layer=self.pm.layer_manager.active,
                          x=dlg.x.value(), y=dlg.y.value(), z=dlg.z.value(), name=f"Ponto {eid}")
            self.entity_store.save(ent)
            if self.pm.journal:
                self.pm.journal.record(JournalOp(op='create', target=eid, before=None, after=ent.to_json()))
            self.reload_entities()

    def action_new_point_click(self):
        if not (self.pm.layer_manager and self.entity_store):
            return
        self.statusBar().showMessage("Modo: Ponto 3D — clique no plano Z=0 (Esc p/ cancelar)")
        self.viewport.start_point_mode(self._on_point_clicked)

    def _on_point_clicked(self, p: Tuple[float,float,float]):
        if not self.entity_store or not self.pm.layer_manager:
            return
        eid = self.entity_store.new_id()
        ent = Point3D(id=eid, type='Point3D', layer=self.pm.layer_manager.active,
                      x=p[0], y=p[1], z=p[2], name=f"Ponto {eid}")
        self.entity_store.save(ent)
        if self.pm.journal:
            self.pm.journal.record(JournalOp(op='create', target=eid, before=None, after=ent.to_json()))
        self.reload_entities()
        self.statusBar().clearMessage()

    def action_new_polyline(self):
        if not (self.pm.layer_manager and self.entity_store):
            return
        dlg = NewPolylineDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            verts = dlg.vertices()
            if len(verts) < 2:
                QMessageBox.warning(self, "Polyline 3D", "Informe ao menos dois vértices.")
                return
            eid = self.entity_store.new_id()
            ent = Polyline3D(id=eid, type='Polyline3D', layer=self.pm.layer_manager.active,
                             vertices=verts, closed=False, name=f"Polyline {eid}")
            self.entity_store.save(ent)
            if self.pm.journal:
                self.pm.journal.record(JournalOp(op='create', target=eid, before=None, after=ent.to_json()))
            self.reload_entities()

    def action_new_polyline_click(self):
        if not (self.pm.layer_manager and self.entity_store):
            return
        self.statusBar().showMessage("Modo: Polyline 3D — clique para adicionar vértices; Enter/d.uplo-clique/direito finaliza; Esc cancela")
        self.viewport.start_polyline_mode(self._on_polyline_clicked)

    def _on_polyline_clicked(self, verts: List[Tuple[float,float,float]]):
        if not self.entity_store or not self.pm.layer_manager:
            return
        eid = self.entity_store.new_id()
        ent = Polyline3D(id=eid, type='Polyline3D', layer=self.pm.layer_manager.active,
                         vertices=verts, closed=False, name=f"Polyline {eid}")
        self.entity_store.save(ent)
        if self.pm.journal:
            self.pm.journal.record(JournalOp(op='create', target=eid, before=None, after=ent.to_json()))
        self.reload_entities()
        self.statusBar().clearMessage()

    def action_new_polyline_table(self):
        if not (self.pm.layer_manager and self.entity_store):
            return
        dlg = PolylineTableDialog(self)
        if dlg.exec() == QtWidgets.QDialog.Accepted:
            verts = dlg.vertices()
            if len(verts) < 2:
                QMessageBox.warning(self, "Polyline 3D", "Informe ao menos dois vértices.")
                return
            eid = self.entity_store.new_id()
            ent = Polyline3D(id=eid, type='Polyline3D', layer=self.pm.layer_manager.active,
                             vertices=verts, closed=False, name=f"Polyline {eid}")
            self.entity_store.save(ent)
            if self.pm.journal:
                self.pm.journal.record(JournalOp(op='create', target=eid, before=None, after=ent.to_json()))
            self.reload_entities()

    # -------- Exibição --------
    def action_reset_view(self):
        self.viewport.camera_dist = 50.0
        self.viewport.rot_x = -30.0
        self.viewport.rot_y = 45.0
        self.viewport.update()

    # -------- Util --------
    def reload_entities(self):
        if not self.entity_store:
            return
        ents = self.entity_store.load_all()
        self.viewport.set_entities(ents)

# ==========================
# Main
# ==========================
def main():
    # Define formato padrão do contexto antes de criar a aplicação/GL widgets
    fmt = QSurfaceFormat()
    fmt.setRenderableType(QSurfaceFormat.OpenGL)
    fmt.setProfile(QSurfaceFormat.CoreProfile)
    fmt.setVersion(3, 3)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

if __name__ == '__main__':
    main()
