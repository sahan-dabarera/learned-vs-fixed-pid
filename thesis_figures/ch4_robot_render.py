"""Ch 4 figure: renderings of the simulated robot (perspective and top view).

Run: python thesis_figures/ch4_robot_render.py   (writes two PNGs to thesis_figures/output/).
Needs an OpenGL context for the MuJoCo renderer.
"""
from pathlib import Path

import mujoco
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "pathfollow" / "core" / "car_model.xml"
OUT = Path(__file__).resolve().parent / "output"
OUT.mkdir(exist_ok=True)

VISUAL = '<mujoco><visual><global offwidth="1600" offheight="1200"/><quality shadowsize="4096"/></visual>'


def main():
    xml = XML.read_text(encoding="utf-8").replace("<mujoco>", VISUAL, 1)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    for _ in range(300):  # let the robot settle on the floor, no commands
        mujoco.mj_step(model, data)
    renderer = mujoco.Renderer(model, 1200, 1600)

    def shot(name, azimuth, elevation, distance, lookat=(0, 0, 0.02)):
        camera = mujoco.MjvCamera()
        camera.lookat[:] = lookat
        camera.azimuth, camera.elevation, camera.distance = azimuth, elevation, distance
        renderer.update_scene(data, camera)
        Image.fromarray(renderer.render()).save(OUT / name)

    shot("robot_persp.png", 135, -25, 0.55)
    shot("robot_top.png", 90, -89.9, 0.45)


if __name__ == "__main__":
    main()
