"""
Import motion tracked camera data into camera in Unreal Engine and render a scene.

This script can only be run in the Python terminal used by Unreal Engine 4.25+.
"""

import unreal
import csv


class capture_frame(object):
    """
    Capture frames class for moving the camera based off of motion tracked footage.
    """

    def __init__(self, camera_actor, camera_tracks):
        unreal.EditorLevelLibrary.editor_set_game_view(True)
        self.camera_actor = camera_actor
        self.positions = (frame_pos for frame_pos in camera_tracks)
        self.frame_index = 0
        self.on_pre_tick = unreal.register_slate_pre_tick_callback(self.capture)

    def capture(self, deltatime):
        """
        Render the scene out of the selected camera viewport for every frame's track.

        :param deltatime: time tick of each frame on the engine's renderer
        :type deltatime: float
        """
        try:
            frame_pos = next(self.positions)
            x = frame_pos[2]
            y = frame_pos[0]
            z = frame_pos[1]
            r1 = frame_pos[3] * 180.0
            r2 = frame_pos[4] * 180.0
            r3 = frame_pos[5] * 180.0
            transformation = unreal.Transform(
                location=[x, y, z], rotation=[r1, r2, r3], scale=[0.0, 0.0, 0.0]
            )
            self.camera_actor.set_actor_transform(transformation, False, False)
            unreal.EditorLevelLibrary.pilot_level_actor(camera_actor)
            unreal.AutomationLibrary.take_high_res_screenshot(
                1920, 1080, "shot" + str(self.frame_index) + ".png"
            )
            unreal.EditorLevelLibrary.eject_pilot_level_actor()
            self.frame_index += 1
        except Exception as error:
            print(error)
            unreal.unregister_slate_pre_tick_callback(self.on_pre_tick)


def get_camera_actor(actors_list):
    """
    Iterate through selected actors to find the camera actor in the scene.
    :param actors_list: list of actors selected and passed to script
    :type actors_list: list[unreal.Actor]
    :return: specified camera actors
    :rtype: unreal.CameraActor
    """
    for actor in actors_list:
        if isinstance(actor, unreal.CameraActor):
            return actor
    return actors_list[0]


camera_tracks = list(csv.reader(open("experimental.csv"), quoting=csv.QUOTE_NONNUMERIC))

selected_actors_list = unreal.EditorLevelLibrary.get_selected_level_actors()
camera_actor = get_camera_actor(selected_actors_list)

task = capture_frame(camera_actor, camera_tracks)
