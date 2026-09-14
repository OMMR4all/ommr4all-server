from django.db import models
from rest_framework import serializers

from database.models.bookstyles import BookStyle

SYMBOL_CLASS_BASE_TYPES = ('note', 'clef', 'accid')


class SymbolClass(models.Model):
    """A symbol class registered at runtime, refining a built-in base class.

    The folder-based PCGTS files reference a class only by `id`; a row that is
    deleted leaves existing annotations intact, they degrade to their base class.
    """
    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=255)
    # null = available in every notation style
    style = models.ForeignKey(BookStyle, null=True, blank=True, on_delete=models.CASCADE,
                              related_name='symbol_classes')
    base_symbol_type = models.CharField(max_length=16)   # SymbolType value
    base_sub_type = models.CharField(max_length=32)      # ClefType/AccidType value, or NoteType as str
    clef_offset = models.IntegerField(null=True, blank=True)   # only meaningful for clefs
    glyph_preset = models.CharField(max_length=32, blank=True, default='')
    svg_path = models.TextField(blank=True, default='')
    svg_path_stroke = models.FloatField(null=True, blank=True)
    color = models.CharField(max_length=32, blank=True, default='')   # CSS colour, '' = user appearance colour
    digit_shortcut = models.IntegerField(null=True, blank=True)
    hidden_by_default = models.BooleanField(default=False)
    order = models.IntegerField(default=0)

    class Meta:
        ordering = ['order', 'id']


class SymbolClassSerializer(serializers.ModelSerializer):
    style = serializers.PrimaryKeyRelatedField(queryset=BookStyle.objects.all(), required=False,
                                               allow_null=True)

    class Meta:
        model = SymbolClass
        fields = ['id', 'name', 'style', 'base_symbol_type', 'base_sub_type', 'clef_offset',
                  'glyph_preset', 'svg_path', 'svg_path_stroke', 'color', 'digit_shortcut',
                  'hidden_by_default', 'order']
