from .unknown import FromUnknownConverter


class FromJaxConverter(FromUnknownConverter):

    def __init__(self, xp_target):
        super().__init__(xp_target)

    def to_numpy(self, array, **kwargs):
        return self.xp_target.asarray(array, **kwargs)

    def to_cupy(self, array, **kwargs):
        import cupy as cp

        return cp.from_dlpack(array)

    # def to_torch(self, array, **kwargs):
    #     return

    def to_jax(self, array, **kwargs):
        # TODO: device handling?
        return array
