"""Dependency-free obstacle contact policy used by the vendored Box2D listener."""
from dataclasses import dataclass
from typing import Iterable


@dataclass
class ObstacleUserData:
    hit: bool = False


def _same_body(left, right) -> bool:
    """Compare Python identity or the native handle of Box2D SWIG proxies."""
    if left is right:
        return True

    left_handle = getattr(left, "this", None)
    right_handle = getattr(right, "this", None)
    return (
        left_handle is not None
        and right_handle is not None
        and left_handle == right_handle
    )


def match_obstacle_body(body_a, body_b, car_bodies: Iterable):
    """Return the obstacle when the other body is the same Box2D vehicle body.

    Box2D may return a fresh SWIG proxy for a native body from a contact, so
    Python object identity alone is not a valid native-body identity check.
    """
    car_bodies = tuple(car_bodies)
    for obstacle_body, other_body in ((body_a, body_b), (body_b, body_a)):
        if (
            isinstance(getattr(obstacle_body, "userData", None), ObstacleUserData)
            and any(_same_body(other_body, car_body) for car_body in car_bodies)
        ):
            return obstacle_body
    return None


def mark_first_obstacle_hit(obstacle_body) -> bool:
    """Mark and report the first hit; every later contact returns False."""
    if obstacle_body is None or obstacle_body.userData.hit:
        return False
    obstacle_body.userData.hit = True
    return True
