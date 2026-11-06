"""MuJoCo window with real GLFW press/release and focus-loss events.

No global keyboard hook and no built-in render shortcuts on movement keys.
This viewer intentionally offers a compact HUD rather than native side panels.
"""
from contextlib import nullcontext
import time
import glfw
import mujoco


class GLFWViewer:
    def __init__(self, model, data, key_callback, status_callback=None):
        self.m=model;self.d=data;self._key_callback=key_callback
        self._status_callback=status_callback
        self._window=None;self._context=None
        if not glfw.init():
            raise RuntimeError('GLFW could not connect to your desktop. Run from a local desktop terminal.')
        try:
            self._window=glfw.create_window(1280,800,'MuJoCo - hold controls',None,None)
            if not self._window:raise RuntimeError('Could not create MuJoCo OpenGL window.')
            glfw.make_context_current(self._window);glfw.swap_interval(0)
            self.cam=mujoco.MjvCamera();self.opt=mujoco.MjvOption();self.perturb=mujoco.MjvPerturb()
            self.user_scn=mujoco.MjvScene(model,maxgeom=max(10000,model.ngeom+100))
            self._context=mujoco.MjrContext(model,mujoco.mjtFontScale.mjFONTSCALE_100)
            mujoco.mjr_setBuffer(mujoco.mjtFramebuffer.mjFB_WINDOW,self._context)
            self._cursor=glfw.get_cursor_pos(self._window)
            self._wall_start=time.monotonic();self._sim_start=float(data.time);self._last_draw=0.0
            glfw.set_key_callback(self._window,self._on_key)
            glfw.set_window_focus_callback(self._window,self._on_focus)
            glfw.set_cursor_pos_callback(self._window,self._on_mouse_move)
            glfw.set_scroll_callback(self._window,self._on_scroll)
        except BaseException:
            self.close();raise

    def _on_key(self,window,key,scancode,action,mods):
        if glfw.KEY_KP_0<=key<=glfw.KEY_KP_9:key=ord('0')+key-glfw.KEY_KP_0
        # Autorepeat must not affect acceleration or retrigger camera/jump toggles.
        if action in (glfw.PRESS,glfw.RELEASE):self._key_callback((key,action))

    def _on_focus(self,window,focused):
        if not focused:self._key_callback((-1,glfw.RELEASE))

    def _on_mouse_move(self,window,x,y):
        last_x,last_y=self._cursor;self._cursor=(x,y)
        left=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_LEFT)==glfw.PRESS
        right=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_RIGHT)==glfw.PRESS
        middle=glfw.get_mouse_button(window,glfw.MOUSE_BUTTON_MIDDLE)==glfw.PRESS
        if not (left or right or middle):return
        _,height=glfw.get_window_size(window)
        if height<=0:return
        shift=(glfw.get_key(window,glfw.KEY_LEFT_SHIFT)==glfw.PRESS or glfw.get_key(window,glfw.KEY_RIGHT_SHIFT)==glfw.PRESS)
        if right:action=mujoco.mjtMouse.mjMOUSE_MOVE_H if shift else mujoco.mjtMouse.mjMOUSE_MOVE_V
        elif left:action=mujoco.mjtMouse.mjMOUSE_ROTATE_H if shift else mujoco.mjtMouse.mjMOUSE_ROTATE_V
        else:action=mujoco.mjtMouse.mjMOUSE_ZOOM
        mujoco.mjv_moveCamera(self.m,action,(x-last_x)/height,(y-last_y)/height,self.user_scn,self.cam)

    def _on_scroll(self,window,xoffset,yoffset):
        mujoco.mjv_moveCamera(self.m,mujoco.mjtMouse.mjMOUSE_ZOOM,0,-.05*yoffset,self.user_scn,self.cam)

    def poll_events(self):
        glfw.poll_events()

    def lock(self):return nullcontext()  # simulation, input and render share one thread

    def is_running(self):
        return self._window is not None and not glfw.window_should_close(self._window)

    def sync(self):
        # Pace physics to actual model time; render independently at up to60Hz.
        now=time.monotonic();deadline=self._wall_start+float(self.d.time)-self._sim_start
        if deadline>now:time.sleep(min(deadline-now,.02))
        elif now-deadline>.25:
            self._wall_start=now;self._sim_start=float(self.d.time)
        now=time.monotonic()
        if now-self._last_draw<1/60:return
        self._last_draw=now
        width,height=glfw.get_framebuffer_size(self._window)
        if width<=0 or height<=0:return
        viewport=mujoco.MjrRect(0,0,width,height)
        mujoco.mjv_updateScene(self.m,self.d,self.opt,self.perturb,self.cam,mujoco.mjtCatBit.mjCAT_ALL,self.user_scn)
        mujoco.mjr_render(viewport,self.user_scn,self._context)
        controls='HOLD 1/2: linear -/+ | HOLD 3/4: yaw +/- | 7: stop | 0: camera | Esc: quit'
        status=self._status_callback() if self._status_callback else ''
        mujoco.mjr_overlay(mujoco.mjtFont.mjFONT_NORMAL,mujoco.mjtGridPos.mjGRID_TOPLEFT,viewport,controls,status,self._context)
        glfw.swap_buffers(self._window)

    def close(self):
        if self._context is not None:
            self._context.free();self._context=None
        if self._window is not None:
            glfw.destroy_window(self._window);self._window=None
        glfw.terminate()

    def __enter__(self):return self
    def __exit__(self,*args):self.close();return False
