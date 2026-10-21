import numpy as np

from motion.background import Background, find_blobs, to_gray
from motion.common import Image
from tests.helpers import RED, scene, textured


def settle(bg: Background, frame: Image, n: int = 15) -> Image:
    mask = np.zeros(1, np.uint8)
    for _ in range(n):
        mask = bg.apply(to_gray(frame), threshold=25)
    return mask


def test_static_scene_gives_empty_mask() -> None:
    assert not settle(Background(), scene()).any()


def test_warmup_frames_are_always_quiet() -> None:
    bg = Background(warmup=5)
    bg.apply(to_gray(scene()), 25)
    assert not bg.apply(to_gray(scene((100, 100, 40, RED))), 25).any()


def test_new_object_shows_up_where_it_is() -> None:
    bg = Background()
    settle(bg, scene())
    mask = bg.apply(to_gray(scene((100, 80, 40, (20, 20, 20)))), 25)
    assert mask[100, 120] == 255
    assert mask[10, 10] == 0


def test_object_that_stays_melts_into_background() -> None:
    bg = Background()
    settle(bg, scene())
    still = scene((100, 80, 40, (20, 20, 20)))
    assert bg.apply(to_gray(still), 25).any()
    assert not settle(bg, still, n=40).any()


def test_object_that_leaves_leaves_no_ghost() -> None:
    bg = Background()
    settle(bg, scene())
    pattern = textured(40)
    for x in range(40, 140, 2):  # walk slowly across the frame
        frame = scene()
        frame[80:120, x : x + 40] = pattern
        bg.apply(to_gray(frame), 25)
    assert not bg.apply(to_gray(scene()), 25).any()


def test_ghost_next_to_a_mover_still_heals() -> None:
    bg = Background()
    pattern = textured(40)
    first = scene()
    first[80:120, 20:60] = pattern
    settle(bg, first)  # in view at start: baked into the background
    for i in range(40):  # then walks right, starting next to its own ghost
        frame = scene()
        frame[80:120, 60 + 2 * i : 100 + 2 * i] = pattern
        mask = bg.apply(to_gray(frame), 25)
    assert not mask[80:120, 20:50].any()  # the old spot is clean again


def test_someone_in_view_at_start_leaves_a_ghost_that_heals() -> None:
    bg = Background()
    settle(bg, scene((100, 80, 40, (20, 20, 20))))  # baked into the background
    assert bg.apply(to_gray(scene()), 25).any()  # they moved away: a ghost
    assert not settle(bg, scene(), n=60).any()  # healed within ~6 s at 10 fps


def test_whole_frame_lighting_change_is_absorbed_not_reported() -> None:
    bg = Background()
    settle(bg, scene(base=60))
    assert not bg.apply(to_gray(scene(base=200)), 25).any()


def test_higher_threshold_ignores_faint_change() -> None:
    faint = scene((100, 80, 40, (150, 150, 150)))  # 30 levels below the 180 base
    low, high = Background(), Background()
    settle(low, scene())
    settle(high, scene())
    assert low.apply(to_gray(faint), 10).any()
    assert not high.apply(to_gray(faint), 60).any()


def test_find_blobs_filters_by_area() -> None:
    mask = np.zeros((240, 320), np.uint8)
    mask[10:20, 10:20] = 255  # 100 px
    mask[100:150, 100:150] = 255  # 2500 px
    assert find_blobs(mask, min_area=500) == [(100, 100, 50, 50)]
    assert len(find_blobs(mask, min_area=50)) == 2
