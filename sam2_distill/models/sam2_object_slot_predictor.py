"""SAM2 video predictor with the learned object-slot propagation head."""

from sam2.sam2_video_predictor import SAM2VideoPredictor

from sam2_distill.models.sam2_object_slots import ObjectSlotModelMixin


class SAM2ObjectSlotVideoPredictor(
    ObjectSlotModelMixin,
    SAM2VideoPredictor,
):
    def __init__(
        self,
        *args,
        object_slot_count: int = 0,
        object_slot_min_objects: int = 4,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self._init_object_slots(
            object_slot_count=object_slot_count,
            object_slot_min_objects=object_slot_min_objects,
        )
