from collections import namedtuple

namedtuple("Camera", ["arg", "name", "intrinsics", "distortion", "_state"])


        self.device_id = device_id
        self.name = name
        self.intrinsics = intrinsics
        self.distortion = None
        self._camera = None


