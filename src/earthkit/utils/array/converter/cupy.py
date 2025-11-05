from .unknown import FromUnknownConverter


class FromCupyConverter(FromUnknownConverter):

    def __init__(self, xp_target):
        super().__init__(xp_target)

    def to_numpy(self, array, **kwargs):
        return array.get()

    def to_cupy(self, array, **kwargs):
        return array

    def to_torch(self, array, **kwargs):
        # TODO: add device handling
        from torch.utils.dlpack import from_dlpack

        return from_dlpack(array.toDlpack())

    def to_jax(self, array, **kwargs):
        # TODO: add device handling
        from jax.dlpack import from_dlpack

        return from_dlpack(array.toDlpack())
