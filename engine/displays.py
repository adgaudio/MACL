'''
Note: all with pyglet/opengl
'''
from typing import Callable, Dict, Tuple, List
import pyglet
import threading


class PygletDisplay(object):
    _scheduled_funcs = {}  # type: Dict[PygletDisplay, Dict[Callable, Callable]]
    _scheduled_batch_graphics = {}  # type: Dict[PygletDisplay, Dict[Callable, Callable]]
    _background_eventloop = None  # type: threading.Thread

    def __repr__(self):
        return '%s<%s>' % (
            self.__class__.__name__, self._create_window_kwargs['caption'])

    def __init__(self, width: int, height: int, window_caption: str='') -> None:
        # delay window creation
        self._create_window_kwargs = dict(
            vsync=False, fullscreen=False, visible=True,
            width=width, height=height,
            caption=window_caption or "%s,%s" % (width, height))

        # a set of background graphics to draw
        self._batches = {}  # type: Dict[Callable, pyglet.graphics.Batch]
        self._batches_order = []  # type: List[Tuple[int, Callable]]

        PygletDisplay._scheduled_batch_graphics[self] = {}
        PygletDisplay._scheduled_funcs[self] = {}

    def _create_window(self) -> None:
        self.window = pyglet.window.Window(**self._create_window_kwargs)
        self.window.set_handler("on_draw", self._on_draw)

        # display frame rate
        if self.window.width >= 120 and self.window.height >= 40:
            self.fps_display = pyglet.window.FPSDisplay(self.window)
        else:
            self.fps_display = None
        # helpful for debugging
        #  self.window.push_handlers(pyglet.window.event.WindowEventLogger())

    def _on_draw(self) -> None:
        self.window.clear()
        #  gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        for _, key in self._batches_order:
            self._batches[key].draw()
        if self.fps_display:
            self.fps_display.draw()

    def run(self, start_event_loop=True, background: bool=False) -> None:
        '''
        Show a Pyglet window and initialize the Pyglet event loop.
        Can be called multiple times.  If background is True,
        run Pyglet's event loop in a background thread.
        '''
        # close existing window if exists (ie from interactive mode)
        # this lets us switch from interactive to non-interactive windows.
        if not hasattr(self, 'window'):
            self._create_window()
        elif self.window.has_exit:
            self._create_window()

        if start_event_loop:
            thread = PygletDisplay._background_eventloop
            if background:
                if thread is None or not thread.is_alive():
                    PygletDisplay._background_eventloop = threading.Thread(
                        target=pyglet.app.run)
                    PygletDisplay._background_eventloop.start()
            elif thread is not None and thread.is_alive():
                # bring background to foreground (becomes blocking)
                thread.join()
            else:  # start event loop if not already running.
                pyglet.app.run()

    def unschedule(self, func=None, displays=None, all_displays=False,
                   graphics=False):
        '''Unschedule function from Pyglet's event loop.

        The following kwargs are mutually exclusive:

          - `func` if given, unschedule that particular function
          - `displays` if given, unschedule all functions from given displays
          - `all_displays` if True, unschedule all functions from all displays
          - if no args given (ie by default) unschedule all funcs that this
          instance previously scheduled via the .schedule(...) method.

        - `graphics` used to denote that this function was scheduled by
        .schedule_batch_graphics(...) rather than .schedule(...)
        '''
        if graphics:
            dct = PygletDisplay._scheduled_batch_graphics
        else:
            dct = PygletDisplay._scheduled_funcs

        if displays is not None:
            fns = [
                sdct.pop(k)
                for disp, sdct in dct.items() for k in list(sdct)
                if disp in displays]
        elif all_displays:
            fns = [sdct.pop(k) for sdct in dct.values() for k in list(sdct)]
        elif func is None:
            fns = [dct[self].pop(k) for k in list(dct[self])]
        else:
            fns = [dct[self].pop(func)]
        for f in fns:
            pyglet.clock.unschedule(f)

    def schedule(self, func, interval: float=None, *args, **kwargs) -> None:
        '''Use Pyglet's event loop to schedule functions that continuously run
        over and over again.  Given function is not expected to directly update
        GL graphics.  For details, see pyglet.clock.schedule and if `interval`
        is defined, pyglet.clock.schedule_interval
        If re-scheduling an already scheduled func, unschedule it first.
        '''

        def _hide_dt(dt, *args, **kwargs):
            func(*args, **kwargs)
        if func in PygletDisplay._scheduled_funcs[self]:
            self.unschedule(func)
        PygletDisplay._scheduled_funcs[self][func] = _hide_dt
        if interval:
            pyglet.clock.schedule_interval(_hide_dt, interval, *args, **kwargs)
        else:
            pyglet.clock.schedule(_hide_dt, *args, **kwargs)

    def schedule_batch_graphics(
            self,
            func: Callable[[pyglet.graphics.Batch, float], None],
            rank: int=10, flush_batch: bool=True, interval: float=None,
            *func_args, **func_kwargs) -> None:
        '''
        Schedule `func` to run over and over again.  `func` should
        add graphics to the given batch for OpenGL to draw on screen.

        When drawing points generated by the func,
        the funcs with lower rank are drawn first,
        and the funcs with highest rank are drawn last.

        - `func` receives 2 arguments.  The first arg is a Batch instance, and
        the func should add OpenGL graphics to the batch for later drawing in a
        window.  The second arg is a time delta from from the start of the
        previous call.
        - `rank` Useful if multiple funcs are scheduled to denote which batches
        to draw first.  Later batches override earlier ones.
        - `interval` How many seconds between calls to func.  Assuming func will
        draw on screen, it doesn't need to be faster than your screen's refresh
        interval, which is probably 1/60.
        '''
        def update_batch(dt):
            if flush_batch:
                b = pyglet.graphics.Batch()
                self._batches[func] = b
            else:
                b = self._batches[func]
            func(b, dt, *func_args, **func_kwargs)
        self._batches[func] = pyglet.graphics.Batch()
        self._batches_order.append((rank, func))
        self._batches_order.sort()

        if func in PygletDisplay._scheduled_batch_graphics[self]:
            self.unschedule(func, graphics=True)
        PygletDisplay._scheduled_batch_graphics[self][func] = update_batch

        if interval:
            pyglet.clock.schedule_interval_soft(update_batch, interval)
        else:
            pyglet.clock.schedule(update_batch)

    def close(self, permanently=True):
        if hasattr(self, 'window'):
            self.window.close()

        self.unschedule()
        self.unschedule(graphics=True)
        if permanently:
            if self in PygletDisplay._scheduled_funcs:
                PygletDisplay._scheduled_funcs.pop(self)
            if self in PygletDisplay._scheduled_batch_graphics:
                PygletDisplay._scheduled_batch_graphics.pop(self)
