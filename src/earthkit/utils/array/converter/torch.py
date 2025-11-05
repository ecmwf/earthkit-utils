from .unknown import FromUnknownConverter


class FromTorchConverter(FromUnknownConverter):

    def __init__(self, xp_target):
        super().__init__(xp_target)

    def to_numpy(self, array, **kwargs):
        return array.numpy()

    def to_cupy(self, array, **kwargs):
        # TODO: add device handling
        # (below only works if tensor device is cuda)
        import cupy as cp

        return cp.from_dlpack(array)

    def to_torch(self, array, **kwargs):
        # TODO: add device handling?
        return array

    # def to_jax(self, array, **kwargs):
    #     return
