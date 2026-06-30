import uuid

import cv2
import numpy as np


class DrawingObject:
    def __init__(
        self,
        kind="circle",
        origin=None,
        placement="top-left",
        size=(100, 100),
        color=(0, 255, 0),
        margin=20,
        thickness=-1,
        label=None,
    ):
        self.id = str(uuid.uuid4())

        self.kind = kind
        self.origin = origin
        self.placement = placement
        self.size = size
        self.color = color
        self.margin = margin
        self.thickness = thickness
        self.label = label

        self.initialized = False

        self.position = None
        self.initial_position = None
        self.center_point = None
        self.bounding_box = None

        # interaction state (useful for gestures)
        self.is_hovered = False
        self.is_selected = False

    # ----------------------------
    # POSITION RESOLUTION
    # ----------------------------
    def resolve_position(self, frame_shape):
        frame_h, frame_w = frame_shape[:2]
        obj_w, obj_h = self.size

        positions = {
            "top-left": (self.margin, self.margin),
            "top": ((frame_w - obj_w) // 2, self.margin),
            "top-right": (frame_w - obj_w - self.margin, self.margin),
            "left": (self.margin, (frame_h - obj_h) // 2),
            "center": ((frame_w - obj_w) // 2, (frame_h - obj_h) // 2),
            "right": (frame_w - obj_w - self.margin, (frame_h - obj_h) // 2),
            "bottom-left": (self.margin, frame_h - obj_h - self.margin),
            "bottom": ((frame_w - obj_w) // 2, frame_h - obj_h - self.margin),
            "bottom-right": (
                frame_w - obj_w - self.margin,
                frame_h - obj_h - self.margin,
            ),
        }

        return positions.get(self.placement, positions["top-left"])

    # ----------------------------
    # UPDATE INTERNAL REFERENCES
    # ----------------------------
    def update_reference_points(self):
        if self.origin is None:
            return

        x, y = self.origin
        w, h = self.size

        self.position = {"x": x, "y": y}

        self.center_point = {
            "x": x + w // 2,
            "y": y + h // 2,
        }

        self.bounding_box = {
            "x1": x,
            "y1": y,
            "x2": x + w,
            "y2": y + h,
        }

    # ----------------------------
    # INITIALIZATION
    # ----------------------------
    def initialize_position(self, frame_shape):
        if self.origin is None:
            self.origin = self.resolve_position(frame_shape)

        self.initial_position = {
            "x": self.origin[0],
            "y": self.origin[1],
        }

        self.update_reference_points()
        self.initialized = True

    # ----------------------------
    # MOVE / RESET
    # ----------------------------
    def move_to(self, x, y):
        self.origin = (int(x), int(y))
        self.update_reference_points()

    def reset_position(self):
        if self.initial_position:
            self.origin = (
                self.initial_position["x"],
                self.initial_position["y"],
            )
            self.update_reference_points()

    # ----------------------------
    # INTERACTION HELPERS
    # ----------------------------
    def contains_point(self, point):
        if not self.bounding_box:
            return False

        px, py = point
        return (
            self.bounding_box["x1"] <= px <= self.bounding_box["x2"]
            and self.bounding_box["y1"] <= py <= self.bounding_box["y2"]
        )

    def set_hover(self, state: bool):
        self.is_hovered = state

    def set_selected(self, state: bool):
        self.is_selected = state

    # ----------------------------
    # GETTERS
    # ----------------------------
    def get_position(self):
        return self.position

    def get_center(self):
        return self.center_point

    def get_bounds(self):
        return self.bounding_box

    def get_size(self):
        return self.size

    def get_id(self):
        return self.id

    # ----------------------------
    # SERIALIZATION
    # ----------------------------
    def to_dict(self):
        return {
            "id": self.id,
            "kind": self.kind,
            "origin": self.origin,
            "placement": self.placement,
            "size": self.size,
            "color": self.color,
            "label": self.label,
            "initialized": self.initialized,
        }

    # optional reverse (useful for restoring state)
    @staticmethod
    def from_dict(data: dict):
        obj = DrawingObject(
            kind=data.get("kind", "circle"),
            origin=data.get("origin"),
            placement=data.get("placement", "top-left"),
            size=tuple(data.get("size", (100, 100))),
            color=tuple(data.get("color", (0, 255, 0))),
            label=data.get("label"),
        )
        obj.initialized = data.get("initialized", False)
        obj.update_reference_points()
        return obj

    # ----------------------------
    # LAYOUT ENGINE
    # ----------------------------
    @staticmethod
    def distribute_evenly(
        drawings,
        frame_shape,
        row="top",
        margin=20,
        padding=10,
    ):
        frame_h, frame_w = frame_shape[:2]

        count = len(drawings)
        if count == 0:
            return drawings  # IMPORTANT FIX

        total_object_width = sum(d.size[0] for d in drawings)

        available_width = frame_w - (margin * 2) - total_object_width

        gap = max(padding, available_width // max(count - 1, 1))

        current_x = margin

        for drawing in drawings:
            obj_w, obj_h = drawing.size

            if row == "top":
                y = margin
            elif row == "center":
                y = (frame_h - obj_h) // 2
            elif row == "bottom":
                y = frame_h - obj_h - margin
            else:
                y = margin

            drawing.origin = (int(current_x), int(y))
            drawing.initial_position = {"x": int(current_x), "y": int(y)}
            drawing.update_reference_points()
            drawing.initialized = True

            current_x += obj_w + gap

        return drawings

    # ----------------------------
    # DRAWING
    # ----------------------------
    def draw(self, frame):
        if not self.initialized or self.origin is None:
            self.initialize_position(frame.shape)

        x, y = self.origin
        w, h = self.size

        frame_h, frame_w = frame.shape[:2]

        # clamp inside frame
        x = max(0, min(int(x), frame_w - w))
        y = max(0, min(int(y), frame_h - h))

        self.origin = (x, y)
        self.update_reference_points()

        # optional highlight when selected/hovered
        stroke = self.thickness
        if self.is_selected:
            stroke = 3

        if self.kind == "circle":
            center = (x + w // 2, y + h // 2)
            radius = min(w, h) // 2

            cv2.circle(frame, center, radius, self.color, stroke)

        elif self.kind == "square":
            side = min(w, h)
            cv2.rectangle(frame, (x, y), (x + side, y + side), self.color, stroke)

        elif self.kind == "rectangle":
            cv2.rectangle(frame, (x, y), (x + w, y + h), self.color, stroke)

        elif self.kind == "triangle":
            points = np.array(
                [
                    [x + w // 2, y],
                    [x, y + h],
                    [x + w, y + h],
                ],
                dtype=np.int32,
            )

            if stroke == -1:
                cv2.fillPoly(frame, [points], self.color)
            else:
                cv2.polylines(frame, [points], True, self.color, stroke)

        # label
        if self.label:
            cv2.putText(
                frame,
                self.label,
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.color,
                2,
            )

        return frame
