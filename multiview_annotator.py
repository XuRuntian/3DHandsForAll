"""Six-view extension of the existing Tk annotator; inherited finger controls."""
import copy
from functools import wraps
import json
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, font
import cv2
import numpy as np
from pose_annotator import pose_annotation_app
from multiview_data import mesh_overlay
from multiview_mano import CalibratedMano
from multiview_layout import MultiViewLayout
from sequence_store import ManoSequence, revision
from image_manifest import ImageSequence
from mano_io import default_parameters


FIELD_LABELS = {'pose': '手指关节', 'global_orient': '整体旋转',
                'transl': '整体位置', 'betas': '手型参数'}


class MultiViewApp(MultiViewLayout, pose_annotation_app):
    def __init__(self, args):
        self.args = args
        self.document = None
        self.project_mode = True
        manifest = getattr(args, 'project', None) or args.multiview
        if not Path(manifest).is_file():
            raise ValueError('请使用 --project 指定统一图像 manifest.json；原始估计包需要先通过独立适配器转换。')
        self.sequence = ImageSequence(manifest)
        default_output = self.sequence.root.parent / 'final'
        self.output = Path(args.output or default_output).resolve()
        self.index = self.sequence.frame_ids.index(str(args.frame_id) if args.frame_id is not None else self.sequence.frame_ids[0])
        self.side = args.hand_side
        self.source = None
        self.pending_render = None
        self.is_debugging = False
        self.joint_selected = 0
        if self.output == self.sequence.root or self.output.is_relative_to(self.sequence.root):
            raise ValueError('final 必须位于图像清单输入目录之外。')
        initial_input = args.input or (self.output / 'mano.npz' if (self.output / 'mano.npz').exists() else None)
        if not initial_input:
            raise ValueError('首次使用需要 --input 指定统一 mano.npz；工具不运行估计或生成默认预测。')
        self.output_revision = revision(self.output)
        self._read(initial_input)
        self.window = tk.Tk()
        self.configure_chinese_ui()
        self.window.title('六视角 MANO 人工修正 — 主相机 ' + self.sequence.master_id)
        self.window.protocol('WM_DELETE_WINDOW', self.close)
        self.build_workspace()
        self.images = self.sequence.images(self.index)
        self.sync_controls()
        self.update_rendered_img()

    def configure_chinese_ui(self):
        # Tk's file picker and standard dialog buttons use its bundled locale.
        self.window.tk.call('package', 'require', 'msgcat')
        self.window.tk.call('::msgcat::mclocale', 'zh_cn')
        available = {name.casefold(): name for name in font.families(self.window)}
        for family in ('Noto Sans CJK SC', 'Microsoft YaHei', 'WenQuanYi Micro Hei',
                       'Droid Sans Fallback', 'song ti', 'fangsong ti'):
            if family.casefold() in available:
                family = available[family.casefold()]
                for name in ('TkDefaultFont', 'TkTextFont', 'TkMenuFont', 'TkHeadingFont'):
                    font.nametofont(name, root=self.window).configure(family=family, size=-16)
                break

    def localize_sliders(self):
        labels = {'rot0': '整体旋转 X', 'rot1': '整体旋转 Y', 'rot2': '整体旋转 Z',
                  'x': '整体位置 X', 'y': '整体位置 Y', 'z': '整体位置 Z'}
        for english, chinese in [('Thumb', '拇指'), ('Index', '食指'), ('Middle', '中指'),
                                 ('Ring', '无名指'), ('Pinky', '小指')]:
            labels.update({english + ' |': chinese + '整体弯曲',
                           english + ' -': chinese + '张开', english + ' o': chinese + '旋转'})
            for i in range(3):
                labels[f'{english} |[{i}]'] = f'{chinese}关节 {i + 1}'
        def visit(widget):
            if isinstance(widget, tk.Scale):
                label = widget.cget('label')
                if label in labels:
                    widget.configure(label=labels[label])
            for child in widget.winfo_children():
                visit(child)
        visit(self.window.panel_sliders)

    def toggle_freeze_restore_text(self, tk_text_var, current_freeze_state):
        tk_text_var.set('恢复记录' if current_freeze_state else '记录当前')
        if not current_freeze_state:
            self.update_rendered_img()

    toggle_freeze_finger_text = toggle_freeze_restore_text

    def select_field(self, label):
        # Localize only the display: persisted JSON parameter keys stay stable.
        key = next(key for key, value in FIELD_LABELS.items() if value == label)
        self.field.set(key)
        limit = {'pose': 44, 'global_orient': 2, 'transl': 2, 'betas': 9}[key]
        self.component.configure(to=limit)
        self.component.delete(0, tk.END)
        self.component.insert(0, '0')
        self.read_exact()

    @staticmethod
    def error_text(exc):
        if isinstance(exc, json.JSONDecodeError):
            return f'文件不是有效的 JSON，请检查第 {exc.lineno} 行、第 {exc.colno} 列。'
        if isinstance(exc, FileNotFoundError):
            return '找不到所需文件，请检查模型或结果文件路径。\n\n详细信息：' + str(exc)
        if isinstance(exc, PermissionError):
            return '没有文件读写权限，请检查文件或保存目录。\n\n详细信息：' + str(exc)
        if isinstance(exc, IndexError):
            return '分量编号超出范围，请选择有效的参数分量。'
        if isinstance(exc, ValueError):
            if any('\u4e00' <= ch <= '\u9fff' for ch in str(exc)):
                return str(exc)
            return '参数或文件内容不符合要求，请检查数值、帧号和坐标系。\n\n详细信息：' + str(exc)
        return '操作未完成。\n\n详细信息：' + str(exc)

    def safe(self, action):
        try:
            action()
        except Exception as exc:
            messagebox.showerror('操作失败', self.error_text(exc), parent=self.window)

    def _new_state(self):
        if self.document is not None:
            state = self.document.record(self.index,self.side)
            self.current_valid = state is not None
            if state is None:
                # A hidden control placeholder only; never rendered or persisted as valid.
                state = default_parameters(self.sequence.frame_ids[self.index],self.side,self.sequence.coordinate_system)
            self.mano_fit_tool = CalibratedMano(self.args.mano_root,state,self.args.device)
            self.original = self.baseline.record(self.index,self.side) or copy.deepcopy(state)
            self.saved_state = json.dumps(state,sort_keys=True)
            self.reset_history()
            return

    def _read(self, path):
        if self.project_mode:
            document = ManoSequence.load(path,self.sequence.coordinate_system,self.sequence.manifest_sha256)
            loaded_revision = document.disk_revision
            document = document.aligned(self.sequence.frame_ids)
            # Validate model availability before changing the active document.
            candidate = document.record(self.index,self.side) or default_parameters(
                self.sequence.frame_ids[self.index],self.side,self.sequence.coordinate_system)
            CalibratedMano(self.args.mano_root,candidate,self.args.device)
            self.document = document
            self.baseline = copy.deepcopy(document)
            self.source = str(Path(path).resolve())
            if Path(path).resolve() == self.output / 'mano.npz':
                self.output_revision = loaded_revision
            self._new_state()
            return

    def sync_controls(self):
        previous = self.mano_fit_tool.suspend_edits
        self.mano_fit_tool.suspend_edits = True
        try:
            self.update_slider_values()
            for var, value in zip(self.shape_vars, self.mano_fit_tool.state['betas']):
                var.set(value)
            self.window.freeze_text_rot_variable.set('恢复记录' if 'rotation' in self.mano_fit_tool.snapshots else '记录当前')
            for i in range(5):
                getattr(self.window, f'freeze_text_fin{i}_variable').set(
                    '恢复记录' if i in self.mano_fit_tool.snapshots else '记录当前')
            for i,(var,entry) in enumerate(getattr(self,'global_entries',[])):
                value = (np.degrees(self.mano_fit_tool.state['global_orient'][i]) if i<3
                         else self.mano_fit_tool.state['transl'][i-3]*1000.)
                var.set(f'{value:.3f}')
            self.window.update_idletasks()
            self.set_controls_enabled()
        finally:
            self.mano_fit_tool.suspend_edits = previous

    def set_controls_enabled(self):
        if not hasattr(self, '_parameter_widgets'):
            self._parameter_widgets = []
            def visit(widget):
                if 'state' in widget.keys() and widget.winfo_class() not in ('TNotebook',):
                    self._parameter_widgets.append((widget, str(widget.cget('state'))))
                for child in widget.winfo_children():
                    visit(child)
            visit(self.window.panel_sliders)
        for widget, original_state in self._parameter_widgets:
            widget.configure(state=original_state if self.can_edit() else 'disabled')

    def set_shape(self, index, value):
        if not self.mano_fit_tool.suspend_edits:
            self.mano_fit_tool.state['betas'][index] = float(value)
            self.update_rendered_img()

    def read_exact(self):
        value = self.mano_fit_tool.state[self.field.get()][int(self.component.get())]
        self.exact_value.delete(0, tk.END)
        self.exact_value.insert(0, str(value))

    def apply_exact(self):
        from mano_io import validate
        state = copy.deepcopy(self.mano_fit_tool.state)
        state[self.field.get()][int(self.component.get())] = float(self.exact_value.get())
        self.mano_fit_tool.state = validate(state)
        self.sync_controls()
        self.update_rendered_img()

    def reset_history(self):
        self.undo_stack, self.redo_stack = [], []
        self.history_state = copy.deepcopy(self.mano_fit_tool.state)
        self.gesture_active = False

    def checkpoint(self):
        if self.document is not None:
            if not self.current_valid:
                return
            self.document.update(self.index,self.side,self.mano_fit_tool.state)
        state = self.mano_fit_tool.state
        if not self.gesture_active and state != self.history_state:
            self.undo_stack.append(self.history_state)
            self.undo_stack = self.undo_stack[-100:]
            self.redo_stack.clear()
            self.history_state = copy.deepcopy(state)

    def begin_gesture(self):
        self.checkpoint()
        self.gesture_active = True

    def end_gesture(self):
        self.gesture_active = False
        self.checkpoint()
        self.update_rendered_img()

    def undo_edit(self):
        self.end_gesture()
        if not self.undo_stack:
            self.status.set('当前帧没有可撤销的修改')
            return
        self.redo_stack.append(copy.deepcopy(self.mano_fit_tool.state))
        self.mano_fit_tool.state = self.undo_stack.pop()
        self.history_state = copy.deepcopy(self.mano_fit_tool.state)
        self.mano_fit_tool.snapshots.clear()
        self.sync_controls()
        self.update_rendered_img()

    def redo_edit(self):
        if not self.redo_stack:
            self.status.set('当前帧没有可重做的修改')
            return
        self.undo_stack.append(copy.deepcopy(self.mano_fit_tool.state))
        self.mano_fit_tool.state = self.redo_stack.pop()
        self.history_state = copy.deepcopy(self.mano_fit_tool.state)
        self.mano_fit_tool.snapshots.clear()
        self.sync_controls()
        self.update_rendered_img()

    def nudge(self, axis, direction, multiplier=1.):
        delta = np.zeros(3)
        delta[axis] = direction * float(self.nudge_step.get()) * multiplier / 1000.
        camera = self.sequence.cameras[self.selected_camera]
        self.mano_fit_tool.state['transl'] = (np.asarray(self.mano_fit_tool.state['transl']) + camera.R.T @ delta).tolist()
        self.sync_controls()
        self.update_rendered_img()

    def rotate_nudge(self, axis, direction):
        # Compose a camera-axis rotation; do not add Euler angles to axis-angle.
        delta = np.zeros(3)
        delta[axis] = np.deg2rad(float(self.rotation_step.get())) * direction
        camera = self.sequence.cameras[self.selected_camera]
        rotation = cv2.Rodrigues(np.asarray(self.mano_fit_tool.state['global_orient']))[0]
        increment = camera.R.T @ cv2.Rodrigues(delta)[0] @ camera.R
        self.mano_fit_tool.state['global_orient'] = cv2.Rodrigues(increment @ rotation)[0].reshape(3).tolist()
        self.sync_controls()
        self.update_rendered_img()

    def keyboard_nudge(self, event, axis, direction):
        focus = self.window.focus_get()
        if focus is not None and focus.winfo_class() in ('Entry','TEntry','Spinbox','TSpinbox',
                                                        'TCombobox','Scale','TScale'):
            return
        self.safe(lambda:self.nudge(axis,direction,.1 if event.state & 1 else 1.))
        return 'break'

    def adjust_global_component(self, index, direction):
        # Use the same master-camera axis-angle components as the adjacent input.
        field, component = ('global_orient',index) if index<3 else ('transl',index-3)
        step = float((self.rotation_step if index<3 else self.nudge_step).get())
        increment = np.deg2rad(step) if index<3 else step/1000.
        self.mano_fit_tool.state[field][component] += direction * increment
        self.sync_controls()
        self.update_rendered_img()

    def adjust_shape(self, index, direction):
        value = self.mano_fit_tool.state['betas'][index] + direction * .01
        self.set_shape(index, float(np.clip(value,-10,10)))
        self.sync_controls()

    def apply_global_value(self, index):
        value = float(self.global_entries[index][0].get())
        if not np.isfinite(value):
            raise ValueError('请输入有限的数值。')
        field, component = ('global_orient',index) if index<3 else ('transl',index-3)
        self.mano_fit_tool.state[field][component] = float(np.deg2rad(value) if index<3 else value/1000.)
        self.sync_controls()
        self.update_rendered_img()

    def jump_to_frame(self):
        fid = self.frame_entry.get().strip()
        if fid not in self.sequence.frame_ids:
            raise ValueError('找不到此帧号，请输入数据中的同步帧编号。')
        self.navigate(self.sequence.frame_ids.index(fid)-self.index)

    def update_rendered_img(self):
        if self.mano_fit_tool.suspend_edits or not hasattr(self, 'views'):
            return
        self.checkpoint()
        if self.pending_render is None:
            self.pending_render = self.window.after(16, self.render)

    def render(self):
        if self.pending_render is not None:
            self.window.after_cancel(self.pending_render)
        self.pending_render = None
        self.vertices = self.mano_fit_tool.vertices() if self.can_edit() else None  # once for all six views
        self.display_images = {}
        for cid, view in self.views.items():
            image = self.view_image(cid, self.sequence.cameras[cid].size, self.overlay.get())
            self.display_images[cid] = image
            view.set_image(image)
            missing = not self.sequence.has_image(self.index,cid)
            self.view_titles[cid].configure(text=cid + (' · 主相机' if cid==self.sequence.master_id else '') + (' · 缺少图像' if missing else ''))
        self.focus_view.set_image(self.display_images[self.selected_camera])
        side = '左手' if self.side=='left' else '右手'
        dirty = json.dumps(self.mano_fit_tool.state,sort_keys=True) != self.saved_state
        self.frame_caption.set(f'帧 {self.sequence.frame_ids[self.index]} / {len(self.sequence.frames)} 帧 · {side}')
        self.status.set(('● 未保存修改' if dirty else '已就绪') + ' · 六视角共享当前三维手'
                        + (' · 铺满模式已裁切边缘' if self.fit_mode.get().startswith('铺满') else ''))
        if not self.can_edit():
            self.status.set('当前帧这只手无有效多视角拟合结果：不显示网格，不能编辑。')
        for i,scale in enumerate(self.global_scales):
            if i<3:
                value = np.degrees(self.mano_fit_tool.state['global_orient'][i])
                suffix = f'{value:.1f}°（轴角）'
            else:
                value = self.mano_fit_tool.state['transl'][i-3]*1000
                suffix = f'{value:.1f} mm'
            scale.configure(label=scale.base_label+' · '+suffix)
            variable,entry = self.global_entries[i]
            if self.window.focus_get() != entry:
                variable.set(f'{value:.3f}')
        for detail in self.details[:]:
            if detail['window'].winfo_exists():
                self.draw_detail(detail)
            else:
                self.details.remove(detail)

    def view_image(self, cid, size, overlay):
        available = not self.project_mode or self.sequence.has_image(self.index,cid)
        if overlay and self.vertices is not None and available:
            return mesh_overlay(self.images[cid], self.sequence.cameras[cid], self.vertices,
                                self.mano_fit_tool.faces, size)
        return cv2.resize(self.images[cid], size)

    def open_detail(self, cid):
        from tkinter import ttk
        from multiview_viewport import ImageViewport
        from multiview_layout import BG, TEXT
        win = tk.Toplevel(self.window)
        win.title('相机 ' + cid + ' — 原图／手部网格')
        win.geometry('1100x760')
        win.configure(bg=BG)
        bar = tk.Frame(win,bg=BG)
        bar.pack(fill='x',padx=12,pady=10)
        detail = dict(window=win,cid=cid,zoom=tk.DoubleVar(value=1.),
                      overlay=tk.BooleanVar(value=self.overlay.get()))
        ttk.Checkbutton(bar,text='显示网格',variable=detail['overlay'],
                        command=lambda:self.draw_detail(detail)).pack(side='left')
        self.ui_button(bar,'适应窗口',lambda:self.reset_detail(detail)).pack(side='left',padx=8)
        self.ui_button(bar,'原始尺寸',lambda:self.native_detail(detail)).pack(side='left')
        tk.Label(bar,text='滚轮缩放 · 拖动平移 · 双击复位',bg=BG,fg=TEXT).pack(side='right')
        detail['canvas'] = ImageViewport(win,on_focus=lambda:self.reset_detail(detail))
        detail['canvas'].pack(fill='both',expand=True,padx=12,pady=(0,12))
        self.details.append(detail)
        self.draw_detail(detail)

    def reset_detail(self, detail):
        detail['zoom'].set(1.)
        detail['canvas'].reset()
        self.draw_detail(detail)

    def native_detail(self, detail):
        view = detail['canvas']
        w,h = self.sequence.cameras[detail['cid']].size
        zoom = 1. / min(max(1,view.winfo_width())/w,max(1,view.winfo_height())/h)
        detail['zoom'].set(zoom)
        view.pan = (0.,0.)
        self.draw_detail(detail)

    def draw_detail(self, detail):
        if not hasattr(self,'vertices'):
            return
        view = detail['canvas']
        # Only an explicit control change overwrites the viewport's wheel zoom.
        if detail.get('last_zoom') != detail['zoom'].get():
            view.zoom = detail['zoom'].get()
            detail['last_zoom'] = view.zoom
        image = self.view_image(detail['cid'],self.sequence.cameras[detail['cid']].size,detail['overlay'].get())
        view.set_image(image)

    def can_edit(self):
        return self.document is None or self.current_valid

    def save(self):
        if self.document is not None:
            self.checkpoint()
            path = self.document.save(self.output,self.output_revision)
            self.output_revision = revision(self.output)
            self.saved_state = json.dumps(self.mano_fit_tool.state,sort_keys=True)
            self.status.set('已保存整段：' + str(path))
            return path
    def stash(self):
        if self.document is not None:
            self.checkpoint()
            if json.dumps(self.mano_fit_tool.state,sort_keys=True) != self.saved_state:
                self.save()
            return

    def navigate(self, delta):
        if delta == 0:
            return
        index = self.index + delta
        if not 0 <= index < len(self.sequence.frames):
            return
        images = self.sequence.images(index)
        self.stash()
        previous = self.index
        self.index = index
        try:
            self._new_state()
        except Exception:
            self.index = previous
            raise
        self.images = images
        self.sync_controls()
        self.update_rendered_img()

    def switch_side(self):
        self.stash()
        previous = self.side
        self.side = 'left' if self.side == 'right' else 'right'
        try:
            self._new_state()
        except Exception:
            self.side = previous
            raise
        self.sync_controls()
        self.update_rendered_img()

    def load(self):
        path = filedialog.askopenfilename(parent=self.window, title='加载 MANO 预测或修正结果',
                                          filetypes=[('统一 MANO 参数文件', '*.npz')])
        if path:
            self.stash()
            self._read(path)
            self.sync_controls()
            self.update_rendered_img()

    def restore(self):
        self.mano_fit_tool.state = copy.deepcopy(self.original)
        self.mano_fit_tool.snapshots.clear()
        self.sync_controls()
        self.update_rendered_img()

    def close(self):
        try:
            self.stash()
        except Exception as exc:
            messagebox.showerror('保存失败，窗口保持打开', self.error_text(exc), parent=self.window)
            return
        self.window.destroy()


# Programmatic slider synchronization must never quantize imported parameters.
# Reuse the original methods while suppressing their queued Scale callbacks.
def _guard_sync(method):
    @wraps(method)
    def guarded(self):
        previous = self.mano_fit_tool.suspend_edits
        self.mano_fit_tool.suspend_edits = True
        try:
            method(self)
            # Only the outer synchronizer drains queued Scale commands, while
            # edits remain suspended. Nested synchronizers must not nest idle loops.
            if not previous:
                self.window.update_idletasks()
        finally:
            self.mano_fit_tool.suspend_edits = previous
    return guarded


for _name in ['update_slider_values', 'update_slider_rot'] + [f'update_slider_values_fin{i}' for i in range(5)]:
    setattr(MultiViewApp, _name, _guard_sync(getattr(pose_annotation_app, _name)))


def _render_after_restore(method):
    def callback(self):
        method(self)
        self.update_rendered_img()
    return callback


for _name in ['button_freeze_rot_callback'] + [f'button_freeze_fin{i}_callback' for i in range(5)]:
    setattr(MultiViewApp, _name, _render_after_restore(getattr(pose_annotation_app, _name)))


# A Scale variable write may invoke its command again during the idle flush.
# Blocking setters alone is insufficient: a finger callback also synchronizes
# sibling sliders, recursively flushing another idle loop. Stop at the callback
# boundary, before executing *any* of that inherited handler's logic.
def _guard_slider_edit(method):
    @wraps(method)
    def callback(self, value):
        if self.mano_fit_tool.suspend_edits or not self.can_edit():
            return
        return method(self, value)
    return callback


for _name in dir(pose_annotation_app):
    if _name.startswith('on_trackbar_mano_'):
        setattr(MultiViewApp, _name, _guard_slider_edit(getattr(pose_annotation_app, _name)))


def _require_valid_hand(method):
    @wraps(method)
    def guarded(self, *args, **kwargs):
        if self.mano_fit_tool.suspend_edits:
            return
        if not self.can_edit():
            self.status.set('当前帧这只手没有有效拟合结果，不能编辑。')
            return
        return method(self, *args, **kwargs)
    return guarded


for _name in ['set_shape', 'apply_exact', 'nudge', 'rotate_nudge', 'adjust_global_component',
              'adjust_shape', 'apply_global_value', 'restore']:
    setattr(MultiViewApp, _name, _require_valid_hand(getattr(MultiViewApp, _name)))
