from .cropper import Cropper


class NoopCropper(Cropper):
    def _content_rect(self):
        return super()._content_rect()
