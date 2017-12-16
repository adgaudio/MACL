import numpy as np
from contextlib import ContextDecorator, ExitStack
from pyglet import gl
import pyglet
from engine.displays import PygletDisplay
from typing import Tuple
from engine.environments import CartesianEnvironment


class cartesian_display(ContextDecorator):
    ''' Register CartesianEnvironment with PygletDisplay so we can visualize
    the environment in a window.

    Handle scaling and refresh rates, and only draw points with information

    >>> with cartesian_display(env) as d:
        d.schedule(agent.next_action, None, env)
    '''
    def __init__(self,
                 env: CartesianEnvironment,
                 refresh_interval: float=1 / 35,
                 scale: Tuple[int, int]=(1, 1),
                 agents: bool=True,
                 visit_counts: bool=True,
                 visit_statistics: bool=True,
                 window_caption: str=None,
                 background: bool=False) -> None:
        if visit_statistics:
            visit_statistics_height = 65
        else:
            visit_statistics_height = 0
        self.__dict__.update(locals())
        self.quads = scale[0] > 1 or scale[1] > 1
        self.window_caption = \
                '%s' % env if window_caption is None else window_caption
        self.display = PygletDisplay(
            width=env.grid_shape[0] * scale[0],
            height=env.grid_shape[1] * scale[1] + visit_statistics_height,
            window_caption=self.window_caption
        )

    def __enter__(self):
        '''Schedule graphics'''
        if self.visit_counts:
            self.display.schedule_batch_graphics(
                self._display_visit_counts,
                rank=1,
                interval=self.refresh_interval,
                env=self.env, scale=self.scale, quads=self.quads
            )
        if self.agents:
            self.display.schedule_batch_graphics(
                self._display_agent_locations,
                rank=2,
                interval=self.refresh_interval,
                env=self.env, scale=self.scale, quads=self.quads)
        if self.visit_statistics:
            self.display.schedule_batch_graphics(
                self._display_visit_statistics,
                rank=3,
                interval=self.refresh_interval,
                env=self.env, scale=self.scale,
                window_caption=self.window_caption)
        self.display.run(False)
        return self.display

    def __exit__(self, exc_type, exc, exc_tb):
        if self.background:
            self.display.run(background=True)
        else:
            self.display.run()  # bring event loop to foreground.
            # clean up
            self.display.close(False)

    @staticmethod
    def _display_visit_counts(batch, dt, env, scale, quads):

        r, c = env.grid_shape
        # don't write points that haven't been visited
        mask = (env._grid_state > 0).reshape(-1)
        points = np.mgrid[:r, :c].swapaxes(1, 2).T.reshape(-1, 2) * scale
        if quads:  # configure for gl_QUADS
            points = points.reshape(-1, 2).repeat(4, axis=0).reshape(-1) + \
                np.tile([0, 0, scale[0], 0, scale[0], scale[1], 0, scale[1]],
                        mask.shape[0])
            points = points[mask.repeat(4 * 2)]
        else:  # configure for gl_POINTS
            points = points.reshape(-1)[mask.repeat(2)]

        colors = (env._grid_state / (env._grid_state.max() + 1) * (255-20) + 20)
        colors[env._grid_state == 0] = 0
        colors = colors.ravel()[mask].astype('uint8').repeat(3)
        if quads:
            colors = colors.reshape((-1, 3)).repeat(4, axis=0).reshape(-1)
        batch.add(
            mask.sum() * (4 if quads else 1),
            gl.GL_QUADS if quads else gl.GL_POINTS, None,
            ('v2i', points), ('c3B', colors))

    @staticmethod
    def _display_agent_locations(batch, dt, env, scale, quads):
        for agent_id, loc in env._agents.items():
            loc = loc * scale
            if quads:  # configure for gl_QUADS
                points = \
                    loc.repeat(4, axis=0).view(np.ndarray).reshape(-1) + \
                    np.tile(
                        [0, 0, scale[0], 0, scale[0], scale[1], 0, scale[1]],
                        loc.shape[0])
            else:  # configure for gl_POINTS
                points = loc.view(np.ndarray).reshape(-1)
            colors = np.tile([
                agent_id % 200 + 55,
                agent_id // 7 % 200 + 55,
                agent_id // 11 % 200 + 55],
                loc.shape[0])
            if quads:
                colors = colors.reshape((-1, 3)).repeat(4, axis=0).reshape(-1)
            batch.add(
                loc.shape[0] * (4 if quads else 1),
                gl.GL_QUADS if quads else gl.GL_POINTS, None,
                ('v2i', points), ('c3B', colors))

    @staticmethod
    def _display_visit_statistics(batch, dt, env, scale, window_caption):
        pyglet.text.Label(
            window_caption + ' ' + \
            ' '.join('%s:%s' % (k, v)
                      for k, v in sorted(env.visit_statistics().items())),
            font_name='Times New Roman',
            font_size=8,
            color=(255, 255, 255, 150),
            x=1, y=env.grid_shape[1] * scale[1] + 1,
            anchor_x='left', anchor_y='bottom',
            multiline=True, width=env.grid_shape[0],
            batch=batch,
        )


class many_displays(ExitStack):
    '''Manages the display of many simultaneous windows

    >>> with many_displays( *[ cartesian_display(env) ] * 2) as displays:
        [d.schedule(agent.next_action, None, env) for d in displays]

    '''
    def __init__(self, *adapters: cartesian_display) -> None:
        super().__init__()
        self.adapters = adapters

    def __enter__(self):
        super().__enter__()

        self.saved_backgrounds = [x.background for x in self.adapters]
        for ad in self.adapters:
            ad.background = True
        displays = tuple(self.enter_context(ad) for ad in self.adapters)
        self.layout_grid_wrap(displays)
        return displays

    def layout_grid_wrap(self, displays: Tuple[PygletDisplay]):
        screen = displays[0].window.display.get_default_screen()
        xmargin, ymargin = 5, 30 + 5
        x, y = xmargin, ymargin
        yt_max = y
        for disp in displays:
            if y > screen.height:
                raise Exception(
                    "Not enough screen real estate to fit all windows")
            disp.window.set_location(x, y)
            x += disp.window.width + xmargin
            yt_max = max(yt_max, y + disp.window.height + ymargin )
            if x + disp.window.width + xmargin > screen.width:
                x, y = xmargin, yt_max + 50  # y accounts for window dialog

    def __exit__(self, *stuff):
        super().__exit__(*stuff)
        for ad, saved_val in zip(self.adapters, self.saved_backgrounds):
            ad.background = saved_val
            if any(not x for x in self.saved_backgrounds):
                PygletDisplay._background_eventloop.join()
