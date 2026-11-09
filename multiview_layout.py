"""Presentation-only layout for the existing MANO editing callbacks."""
import tkinter as tk
from tkinter import ttk, font
from multiview_viewport import ImageViewport

BG, PANEL, CARD, TEXT, MUTED, ACCENT = '#0c121b', '#16212f', '#1e2c3c', '#e5edf5', '#9cacc0', '#46cbbb'


class MultiViewLayout:
    def build_workspace(self):
        self.window.configure(bg=BG)
        self.window.minsize(1040, 720)
        self.window.geometry('1440x940')
        style = ttk.Style(self.window)
        style.theme_use('clam')
        style.configure('.', background=PANEL, foreground=TEXT, fieldbackground=CARD,
                        bordercolor=CARD, lightcolor=CARD, darkcolor=CARD)
        style.configure('TButton', padding=(10, 7), background=CARD, borderwidth=0)
        style.map('TButton', background=[('active', '#30465c')])
        style.configure('Accent.TButton', background=ACCENT, foreground=BG)
        style.map('Accent.TButton', background=[('active', '#72ded1')])
        style.configure('Display.Toolbutton', padding=(10,5), background=CARD, foreground=TEXT)
        style.map('Display.Toolbutton', background=[('selected', ACCENT), ('active', '#30465c')],
                  foreground=[('selected', BG)])
        style.configure('TNotebook', background=PANEL, borderwidth=0)
        style.configure('TNotebook.Tab', padding=(10, 8), background=CARD)
        style.map('TNotebook.Tab', background=[('selected', ACCENT)], foreground=[('selected', BG)])
        style.configure('TCombobox', padding=4, arrowcolor=TEXT, selectbackground=CARD, selectforeground=TEXT)
        style.map('TCombobox', fieldbackground=[('readonly', CARD)], foreground=[('readonly', TEXT)], selectbackground=[('readonly', CARD)])
        style.configure('Vertical.TScrollbar', background=CARD, troughcolor=PANEL, arrowcolor=MUTED, borderwidth=0, width=12)
        style.configure('TCheckbutton', background=BG, indicatorbackground=CARD, indicatorforeground=ACCENT)
        style.map('TCheckbutton', indicatorbackground=[('selected', ACCENT)], background=[('active', BG)])
        self.window.option_add('*TCombobox*Listbox.background', CARD)
        self.window.option_add('*TCombobox*Listbox.foreground', TEXT)
        self.window.option_add('*Menu.background', CARD)
        self.window.option_add('*Menu.foreground', TEXT)
        self.status = tk.StringVar()
        self.frame_caption = tk.StringVar()
        self.overlay = tk.BooleanVar(value=True)
        self.display_mode = tk.StringVar(value='mesh')
        self.transparency = tk.DoubleVar(value=25.)
        self.transparency_caption = tk.StringVar(value='25%')
        self.layout_mode = tk.StringVar(value='专注视图')
        self.fit_mode = tk.StringVar(value='完整显示')
        self.selected_camera = self.sequence.master_id
        self.details = []
        header = tk.Frame(self.window, bg=BG)
        header.pack(fill='x', padx=18, pady=(14, 8))
        self.title_font = font.nametofont('TkDefaultFont',root=self.window).copy()
        self.title_font.configure(size=-20,weight='bold')
        tk.Label(header, text='MANO  /  多视角修正', bg=BG, fg=TEXT,
                 font=self.title_font).pack(side='left')
        tk.Label(header, text='共享三维手 · 逐帧审核', bg=BG, fg=MUTED).pack(side='left', padx=16)
        self.ui_button(header, '保存修正  Ctrl+S', self.save, accent=True).pack(side='right')
        self.ui_button(header, '加载结果', self.load).pack(side='right', padx=8)
        nav = tk.Frame(self.window, bg=PANEL)
        nav.pack(fill='x', padx=16, pady=(0, 10))
        self.ui_button(nav, '‹ 上一帧', lambda: self.navigate(-1)).pack(side='left', padx=(8,4), pady=7)
        self.ui_button(nav, '下一帧 ›', lambda: self.navigate(1)).pack(side='left')
        tk.Label(nav, textvariable=self.frame_caption, bg=PANEL, fg=TEXT).pack(side='left', padx=14)
        tk.Label(nav, text='跳到帧', bg=PANEL, fg=MUTED).pack(side='left')
        self.frame_entry = ttk.Entry(nav, width=7)
        self.frame_entry.pack(side='left', padx=6)
        self.frame_entry.bind('<Return>', lambda e: self.safe(self.jump_to_frame))
        self.ui_button(nav, '跳转', self.jump_to_frame).pack(side='left')
        self.ui_button(nav, '切换左／右手', self.switch_side).pack(side='right', padx=8)
        self.ui_button(nav, '重做', self.redo_edit).pack(side='right', padx=3)
        self.ui_button(nav, '撤销', self.undo_edit).pack(side='right', padx=3)
        body = tk.PanedWindow(self.window, orient='horizontal', bg=BG, bd=0,
                              sashwidth=10, sashpad=2, opaqueresize=True)
        self.workspace_split = body
        body.pack(fill='both', expand=True, padx=16)
        sidebar = tk.Frame(body, bg=PANEL, width=480)
        self.editor_panel = sidebar
        body.add(sidebar, minsize=380, width=480, stretch='never')
        sidebar.grid_propagate(False)
        sidebar.rowconfigure(1, weight=1)
        sidebar.columnconfigure(0, weight=1)
        title = tk.Frame(sidebar, bg=PANEL)
        title.grid(row=0, column=0, sticky='ew', padx=12, pady=10)
        tk.Label(title, text='编辑当前手', bg=PANEL, fg=TEXT, font=self.title_font).pack(side='left')
        self.ui_button(title, '恢复载入值', self.restore).pack(side='right')
        self.control_scroll = tk.Canvas(sidebar, bg=PANEL, highlightthickness=0, width=450)
        self.control_scroll.grid(row=1, column=0, sticky='nsew')
        scrollbar = ttk.Scrollbar(sidebar, command=self.control_scroll.yview)
        scrollbar.grid(row=1, column=1, sticky='ns')
        self.control_scroll.configure(yscrollcommand=scrollbar.set)
        controls = tk.Frame(self.control_scroll, bg=PANEL)
        control_item = self.control_scroll.create_window(0,0,window=controls,anchor='nw')
        def update_scroll_region(event=None):
            region = self.control_scroll.bbox('all')
            self.control_scroll.configure(scrollregion=region)
            if region and region[3] <= self.control_scroll.winfo_height():
                scrollbar.grid_remove()
                self.control_scroll.yview_moveto(0)
            else:
                scrollbar.grid()
        controls.bind('<Configure>', update_scroll_region)
        self.control_scroll.bind('<Configure>', lambda e: self.control_scroll.itemconfigure(control_item,width=e.width))
        self.window.panel_top_ver = controls
        self.window.panel_sliders = None
        self.init_sliders()  # same widget variables, callbacks and MANO mapping
        self.localize_sliders()
        self.window.slider_joint.pack_forget()  # 2D joint-selection is inactive here
        panel = self.window.panel_sliders
        groups = [w for w in panel.winfo_children() if isinstance(w, tk.PanedWindow)]
        for group in groups:
            group.pack_forget()
        notebook = ttk.Notebook(panel)
        self.controls_notebook = notebook
        notebook.pack(fill='x', padx=8)
        tabs = {}
        for name in ('整体', '手指', '手型', '精确'):
            tabs[name] = tk.Frame(notebook, bg=PANEL)
            notebook.add(tabs[name], text=name)
        groups[0].pack(in_=tabs['整体'], fill='x', pady=8)
        groups[0].lift(notebook)
        self.nudge_step = tk.StringVar(value='1')
        self.rotation_step = tk.StringVar(value='1')
        self.nudge_caption = tk.StringVar()
        self.inline_buttons = []
        self.global_scales = []
        self.global_entries = []
        for group in groups:
            for widget in group.winfo_children():
                widget.pack_forget()
                widget.pack(side='top', fill='x', padx=12, pady=8)
                if isinstance(widget, tk.Scale):
                    widget.configure(showvalue=False, width=18, sliderlength=34, bd=0,
                                     highlightthickness=0, relief='flat', bg=PANEL,
                                     fg=TEXT, troughcolor=CARD, activebackground=ACCENT)
                    widget.base_label = widget.cget('label')
                    if group == groups[0]:
                        index = len(self.global_scales)
                        self.global_scales.append(widget)
                        # Keep the original Scale/callback, adding a direct numeric input.
                        row = tk.Frame(group,bg=PANEL)
                        widget.pack_forget()
                        row.pack(fill='x',padx=10,pady=6)
                        widget.pack(in_=row,side='left',fill='x',expand=True)
                        widget.lift(row)
                        value = tk.StringVar()
                        entry = ttk.Entry(row,textvariable=value,width=8,justify='right')
                        entry.pack(side='right',padx=(8,0),pady=(12,0))
                        entry.bind('<Return>',lambda e,i=index:self.safe(lambda:self.apply_global_value(i)))
                        self.global_entries.append((value,entry))
                        fine = tk.Frame(group,bg=PANEL)
                        fine.pack(fill='x',padx=12,pady=(0,4))
                        step = self.rotation_step if index < 3 else self.nudge_step
                        self.inline_step_row(fine, step, '度' if index < 3 else '毫米',
                                             lambda d,i=index:self.adjust_global_component(i,d))
                    else:
                        fine = tk.Frame(group,bg=PANEL)
                        fine.pack(fill='x',padx=12,pady=(0,4))
                        tk.Label(fine,text='微调 · 每次 1 格',bg=PANEL,fg=MUTED).pack(side='left')
                        self.inline_pair(fine, lambda d,w=widget:w.set(w.get()+d))
                elif isinstance(widget, tk.Button):
                    widget.configure(bg=CARD, fg=TEXT, activebackground=ACCENT,
                                     activeforeground=BG, bd=0, pady=6)
            group.configure(bg=PANEL, bd=0)
        panel.configure(bg=PANEL, bd=0)
        tk.Label(tabs['整体'],text='右侧数值框可直接输入，回车应用。\n旋转单位：度（轴角分量）；位置单位：毫米。',
                 bg=PANEL,fg=MUTED,justify='left').pack(side='bottom',padx=12,pady=12)
        self.finger_choice = tk.StringVar(value='食指')
        fingers = ['拇指','食指','中指','无名指','小指']
        finger_bar = tk.Frame(tabs['手指'],bg=PANEL)
        finger_bar.pack(fill='x',padx=8,pady=12)
        self.finger_buttons = {}
        def select_finger(name):
            self.finger_choice.set(name)
            for group in groups[1:]: group.pack_forget()
            selected = groups[1+fingers.index(name)]
            selected.pack(in_=tabs['手指'], fill='x')
            selected.lift(notebook)
            for finger,button in self.finger_buttons.items():
                button.configure(style='Accent.TButton' if finger==name else 'TButton')
        self.choose_finger = select_finger
        for i,name in enumerate(fingers):
            button=self.ui_button(finger_bar,name,lambda n=name:select_finger(n))
            button.grid(row=0,column=i,sticky='ew',padx=2)
            finger_bar.columnconfigure(i,weight=1)
            self.finger_buttons[name]=button
        select_finger('食指')
        tk.Label(tabs['手指'], text='先调整体弯曲，再微调各关节。', bg=PANEL, fg=MUTED).pack(side='bottom', pady=10)
        tk.Label(tabs['手型'], text='手型由当前手整段共享，修改会影响所有帧。', bg=PANEL, fg=MUTED, wraplength=330).pack(padx=8,pady=8)
        self.shape_vars = []
        for i in range(10):
            var = tk.DoubleVar(value=self.mano_fit_tool.state['betas'][i])
            self.shape_vars.append(var)
            scale = tk.Scale(tabs['手型'], label=f'手型参数 {i+1}', from_=-10, to=10, resolution=.01,
                     variable=var, orient='horizontal', width=16, sliderlength=34,
                     bg=PANEL, fg=TEXT, troughcolor=CARD, bd=0, highlightthickness=0,
                     activebackground=ACCENT, command=lambda v,j=i:self.set_shape(j,v))
            scale.pack(fill='x', padx=8)
            fine = tk.Frame(tabs['手型'],bg=PANEL)
            fine.pack(fill='x',padx=12,pady=(0,8))
            tk.Label(fine,text='微调 · 每次 0.01',bg=PANEL,fg=MUTED).pack(side='left')
            self.inline_pair(fine,lambda d,j=i:self.adjust_shape(j,d))
        exact = tabs['精确']
        from multiview_annotator import FIELD_LABELS
        self.field = tk.StringVar(value='pose')
        self.field_label = tk.StringVar(value=FIELD_LABELS['pose'])
        tk.Label(exact,text='参数类型',bg=PANEL,fg=MUTED).pack(anchor='w',padx=8,pady=(12,4))
        choose = ttk.Combobox(exact,textvariable=self.field_label,values=list(FIELD_LABELS.values()),state='readonly')
        choose.pack(fill='x',padx=8)
        choose.bind('<<ComboboxSelected>>',lambda e:self.select_field(self.field_label.get()))
        tk.Label(exact,text='分量编号（从 0 开始）',bg=PANEL,fg=MUTED).pack(anchor='w',padx=8,pady=(12,4))
        self.component = ttk.Spinbox(exact,from_=0,to=44,width=5)
        self.component.set('0')
        self.component.pack(fill='x',padx=8)
        tk.Label(exact,text='数值 · 位置用米，旋转用弧度',bg=PANEL,fg=MUTED).pack(anchor='w',padx=8,pady=(12,4))
        self.exact_value = ttk.Entry(exact)
        self.exact_value.pack(fill='x',padx=8)
        self.ui_button(exact,'读取当前值',self.read_exact).pack(fill='x',padx=8,pady=(12,4))
        self.ui_button(exact,'应用数值',self.apply_exact,accent=True).pack(fill='x',padx=8)
        def fit_controls(event=None):
            selected = self.window.nametowidget(notebook.select())
            notebook.configure(height=selected.winfo_reqheight())
        notebook.bind('<<NotebookTabChanged>>',lambda e:self.window.after_idle(fit_controls))
        self.window.after_idle(fit_controls)

        self.bind_control_gestures(controls)
        tk.Label(controls,text='拖动中间分隔线，可调整编辑区宽度。',bg=PANEL,fg=MUTED).pack(pady=12)
        stage = tk.Frame(body,bg=BG)
        body.add(stage,minsize=560,stretch='always')
        stage.columnconfigure(0,weight=1)
        stage.rowconfigure(2,weight=1)
        viewbar = tk.Frame(stage,bg=BG)
        viewbar.grid(row=0,column=0,sticky='ew',pady=(0,8))
        modes = ttk.Combobox(viewbar,textvariable=self.layout_mode,values=['六视图','专注视图'],width=8,state='readonly')
        modes.pack(side='left')
        modes.bind('<<ComboboxSelected>>',lambda e:self.arrange_views())
        fits = ttk.Combobox(viewbar,textvariable=self.fit_mode,values=['完整显示','铺满（裁切）'],width=9,state='readonly')
        fits.pack(side='left',padx=7)
        fits.bind('<<ComboboxSelected>>',lambda e:self.change_fit())
        ttk.Checkbutton(viewbar,text='叠加',variable=self.overlay,command=self.update_display).pack(side='left')
        self.ui_button(viewbar,'复位缩放',self.reset_view).pack(side='right')
        self.ui_button(viewbar,'原图窗口',lambda:self.open_detail(self.selected_camera)).pack(side='right',padx=6)
        displaybar = tk.Frame(stage,bg=BG)
        displaybar.grid(row=1,column=0,sticky='ew',pady=(0,8))
        self.build_display_controls(displaybar)
        self.view_grid = tk.Frame(stage,bg=BG)
        self.view_grid.grid(row=2,column=0,sticky='nsew')
        self.views, self.view_cards, self.view_titles = {}, {}, {}
        for cid in self.sequence.cameras:
            card = tk.Frame(self.view_grid,bg=CARD,highlightthickness=1,highlightbackground=CARD)
            self.view_cards[cid] = card
            label = tk.Label(card,text=cid + (' · 主相机' if cid==self.sequence.master_id else ''),bg=CARD,fg=MUTED,anchor='w',padx=10,pady=5)
            label.pack(fill='x')
            label.bind('<Button-1>',lambda e,c=cid:self.select_camera(c))
            self.view_titles[cid] = label
            view = ImageViewport(card,lambda c=cid:self.select_camera(c),lambda c=cid:self.focus_camera(c))
            view.pack(fill='both',expand=True)
            self.views[cid] = view
        self.focus_card = tk.Frame(self.view_grid,bg=CARD)
        self.focus_title = tk.Label(self.focus_card,bg=CARD,fg=TEXT,anchor='w',padx=10,pady=6)
        self.focus_title.pack(fill='x')
        self.focus_view = ImageViewport(self.focus_card,on_focus=self.leave_focus)
        self.focus_view.pack(fill='both',expand=True)
        self.arrange_views()
        footer = tk.Frame(self.window,bg=BG)
        footer.pack(fill='x',padx=18,pady=9)
        tk.Label(footer,textvariable=self.status,bg=BG,fg=ACCENT,anchor='w').pack(side='left')
        tk.Label(footer,text='单击选相机 · 双击放大 · 滚轮缩放 · 拖动平移',bg=BG,fg=MUTED).pack(side='right')
        self.window.bind('<Control-s>',lambda e:self.safe(self.save))
        self.window.bind('<Control-z>',lambda e:self.safe(self.undo_edit))
        self.window.bind('<Control-y>',lambda e:self.safe(self.redo_edit))
        self.window.bind('<Escape>',lambda e:self.leave_focus())
        for key,axis,direction in [('Left',0,-1),('Right',0,1),('Up',1,-1),
                                   ('Down',1,1),('Prior',2,-1),('Next',2,1)]:
            self.window.bind('<'+key+'>',lambda e,a=axis,d=direction:self.keyboard_nudge(e,a,d))

    def build_display_controls(self, parent):
        for label, mode in [('MANO 网格', 'mesh'), ('21 点骨架', 'skeleton')]:
            ttk.Radiobutton(parent,text=label,variable=self.display_mode,value=mode,
                            style='Display.Toolbutton',command=self.update_display).pack(side='left',padx=(0,3))
        tk.Label(parent,text='透明度',bg=BG,fg=TEXT).pack(side='left',padx=(12,5))
        tk.Scale(parent,from_=0,to=100,resolution=1,variable=self.transparency,
                 orient='horizontal',showvalue=False,length=110,width=14,sliderlength=24,
                 bg=BG,fg=TEXT,troughcolor=CARD,activebackground=ACCENT,
                 highlightthickness=0,bd=0,command=self.update_display).pack(side='left',fill='x',expand=True)
        tk.Label(parent,textvariable=self.transparency_caption,bg=BG,fg=TEXT,
                 width=5,anchor='e').pack(side='left')

    def ui_button(self, parent, text, action, accent=False):
        return ttk.Button(parent,text=text,style='Accent.TButton' if accent else 'TButton',
                          command=lambda:self.safe(action))

    def inline_pair(self, parent, action):
        buttons = []
        for text, direction in [('−',-1),('＋',1)]:
            button = self.repeat_button(parent,text,lambda d=direction:action(d))
            button.configure(width=3,padx=8,pady=3)
            button.pack(side='left',padx=(8,0))
            buttons.append(button)
        self.inline_buttons.append(buttons)

    def inline_step_row(self, parent, step, unit, action):
        tk.Label(parent,text='微调',bg=PANEL,fg=MUTED).pack(side='left')
        ttk.Combobox(parent,textvariable=step,values=['0.1','0.5','1','5','10'],
                     state='readonly',width=4).pack(side='left',padx=6)
        tk.Label(parent,text=unit,bg=PANEL,fg=MUTED).pack(side='left')
        self.inline_pair(parent,action)

    def repeat_button(self,parent,text,action):
        button=tk.Button(parent,text=text,bg=CARD,fg=TEXT,activebackground=ACCENT,
                         activeforeground=BG,bd=0,padx=16,pady=10,width=5,
                         repeatdelay=350,repeatinterval=90,command=lambda:self.safe(action))
        button.bind('<ButtonPress-1>',lambda e:self.begin_gesture(),add='+')
        button.bind('<ButtonRelease-1>',lambda e:self.end_gesture(),add='+')
        return button

    def bind_control_gestures(self, parent):
        for widget in parent.winfo_children():
            if isinstance(widget, tk.Scale):
                widget.bind('<ButtonPress-1>',lambda e:self.begin_gesture(),add='+')
                widget.bind('<ButtonRelease-1>',lambda e:self.end_gesture(),add='+')
            widget.bind('<Button-4>',lambda e:self.control_scroll.yview_scroll(-2,'units'),add='+')
            widget.bind('<Button-5>',lambda e:self.control_scroll.yview_scroll(2,'units'),add='+')
            widget.bind('<MouseWheel>',lambda e:self.control_scroll.yview_scroll(-2 if e.delta>0 else 2,'units'),add='+')
            self.bind_control_gestures(widget)

    def arrange_views(self):
        for card in self.view_cards.values():card.grid_forget()
        self.focus_card.grid_forget()
        focused = self.layout_mode.get()=='专注视图'
        for i in range(6):self.view_grid.columnconfigure(i,weight=1 if focused or i<3 else 0,uniform='views' if focused or i<3 else '')
        self.view_grid.rowconfigure(0,weight=1, minsize=0)
        self.view_grid.rowconfigure(1,weight=0 if focused else 1,minsize=118 if focused else 0)
        if focused:self.focus_card.grid(row=0,column=0,columnspan=6,sticky='nsew',pady=(0,8))
        for i,(cid,card) in enumerate(self.view_cards.items()):
            card.grid(row=1 if focused else i//3,column=i if focused else i%3,sticky='nsew',padx=3,pady=3)
        self.select_camera(self.selected_camera)

    def select_camera(self, cid):
        if self.selected_camera != cid:
            self.focus_view.reset()
        self.selected_camera = cid
        for name,card in self.view_cards.items():
            card.configure(highlightbackground=ACCENT if cid==name else CARD)
            self.view_titles[name].configure(fg=ACCENT if cid==name else MUTED)
        self.nudge_caption.set('相机方向微调 · '+cid)
        self.focus_title.configure(text=cid+' · 专注视图（双击或 Esc 返回六视图）')
        if getattr(self,'display_images',None):
            self.focus_view.set_image(self.display_images[cid])

    def focus_camera(self, cid):
        self.layout_mode.set('专注视图')
        self.focus_view.reset()
        self.selected_camera = cid
        self.arrange_views()

    def leave_focus(self):
        self.layout_mode.set('六视图')
        self.arrange_views()

    def change_fit(self):
        mode = 'cover' if self.fit_mode.get().startswith('铺满') else 'fit'
        for view in [*self.views.values(),self.focus_view]:view.set_mode(mode)
        if mode=='cover':self.status.set('铺满画面：保持比例，边缘可能裁切。完整显示可查看全部内容。')

    def reset_view(self):
        (self.focus_view if self.layout_mode.get()=='专注视图' else self.views[self.selected_camera]).reset()
