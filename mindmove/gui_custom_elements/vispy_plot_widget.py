from PySide6.QtWidgets import QWidget, QSizePolicy
from vispy import scene
from vispy.scene.visuals import Axis
from vispy.scene import Line
import numpy as np
from vispy import gloo
from vispy import app
import math
from typing import Union


class VispyPlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        # Plot Widget properties
        self.grid = None
        self.canvas = None
        self.camera = None
        self.title_label = None
        self.title_widget = None
        self.yaxis = None
        self.xaxis = None
        self.xlabel = None
        self.ylabel = None
        self.configured = False

        self.right_padding = None
        self.left_padding = None
        self.bottom_padding = None
        self.view = None

        # Plot details
        self.plot_data = None
        self.plot_xdata = None
        self.line = None
        self.vertexes = None
        self.lines = None
        self.color = None
        self.connect_vertexes = None
        self.display_time = None
        self.sampling_frequency = None
        self.fread = None
        self.frame_len = None

        self.scene = scene.SceneCanvas(
            parent=self, bgcolor="gray", resizable=True, show=True
        )
        self.scene.unfreeze()
        self.grid = self.scene.central_widget.add_grid(spacing=0, margin=10)
        self.central_widget = self.scene.native
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def configure_lines_plot(
        self,
        display_time=10,
        fs=256,
        lines=320,
        color_map=None,
    ) -> None:
        if self.configured:
            return
        self.view = self.grid.add_view(row=0, col=0, camera="panzoom")
        self.camera = self.view.camera
        self.camera.interactive = False

        self.display_time = display_time
        self.sampling_frequency = fs
        self.vertexes = self.display_time * self.sampling_frequency
        self.lines = lines

        self.reset_data()
        self.define_colors(color_map=color_map)
        self.define_connect()

        self.camera.set_range(
            x=[-0.01, self.plot_data[-1, 0]],
            y=[-0.1, 1.1],
        )

        self.line = scene.Line(
            pos=self.plot_data,
            color=self.color,
            parent=self.view.scene,
            connect=self.connect_vertexes,
            width=1,
            # method="agg",
        )
        self.configured = True

    def reset_data(self):
        self.plot_data = np.zeros((self.lines * self.vertexes, 2))
        self.plot_data = self.plot_data.reshape(self.lines, self.vertexes, 2)
        for i, line in enumerate(self.plot_data):
            line[:, 1] += i * 1 / (self.lines)
        self.plot_data = self.plot_data.reshape(self.lines * self.vertexes, 2)
        self.plot_xdata = (
            np.linspace(0, self.vertexes, self.vertexes) / self.sampling_frequency
        )
        self.plot_data[:, 0] = np.tile(self.plot_xdata, self.lines)

    def reset_trajectory(self):
        self.feedback_data = np.zeros((self.vertexes // 2, 2))
        self.trajectory_xdata = (
            np.linspace(0, self.vertexes // 2, self.vertexes // 2)
            / self.sampling_frequency
        )
        self.feedback_data[:, 0] = self.trajectory_xdata

    def define_connect(self):
        self.connect_vertexes = np.empty(
            (self.lines * self.vertexes - self.lines * 1, 2)
        )
        for line in range(self.lines):
            self.connect_vertexes[
                line * self.vertexes - line : self.vertexes * (line + 1) - (line + 1), 0
            ] = np.arange(self.vertexes * line, (line + 1) * self.vertexes - 1)

            self.connect_vertexes[
                line * self.vertexes - line : self.vertexes * (line + 1) - (line + 1), 1
            ] = np.arange(self.vertexes * line + 1, (line + 1) * self.vertexes)

    def define_colors(self, color_map=None):
        colors = []
        if color_map is None:
            colors = np.repeat(np.zeros((self.lines, 3)), self.vertexes, axis=0)
        else:
            for color in color_map:
                colors.append(color)
            colors = np.array(colors)
            colors = colors.reshape(-1, 4) / 255
            colors = np.repeat(colors, self.vertexes, axis=0)
        self.color = colors

    def resizeEvent(self, event) -> None:
        width = self.size().width()
        height = self.size().height()
        self.scene.size = (width, height)

    def set_camera_range(
        self, x_range: tuple[float, float], y_range: tuple[float, float]
    ) -> None:
        self.camera.set_range(x=x_range, y=y_range)

    def set_plot_data(self, data: np.ndarray) -> None:
        frame_len = data.shape[1] if len(data.shape) == 2 else data.shape[0]
        data = data.reshape(self.lines, frame_len) / (self.lines)
        for i in range(data.shape[0]):
            data[i] = data[i] + i / (self.lines)
        plot_data = self.plot_data.copy()
        plot_data = plot_data.reshape(self.lines, self.vertexes, 2)
        plot_data[:, :-frame_len, 1] = plot_data[:, frame_len:, 1]
        plot_data[:, -frame_len:, 1] = data
        self.plot_data = plot_data.reshape(-1, 2)
        self.line.set_data(self.plot_data, self.color, connect=self.connect_vertexes)
        self.scene.update()

    def refresh_plot(self):
        self.scene.central_widget.remove_widget(self.grid)
        self.grid = self.scene.central_widget.add_grid(spacing=0, margin=10)

        self.current_plot_configuration = None
        self.configured = False

    def measure_fps(self):
        self.scene.measure_fps()
