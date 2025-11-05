from .unknown import FromUnknownConverter


class FromNumpyConverter(FromUnknownConverter):

    def __init__(self, xp_target):
        super().__init__(xp_target)

    def to_numpy(self, array, **kwargs):
        return array

    def to_cupy(self, array, **kwargs):
        return self.xp_target.array(array, **kwargs)

    def to_torch(self, array, **kwargs):
        # TODO: add device handling
        return self.xp_target.from_numpy(array, **kwargs)

    def to_jax(self, array, **kwargs):
        # TODO: add device handling
        return self.xp_target.array(array, **kwargs)
