from django.db import models
from rest_framework import serializers

DEFAULT_BOOK_STYLE = 'french14'


class BookStyle(models.Model):
    id = models.CharField(max_length=255, primary_key=True)
    name = models.CharField(max_length=255, unique=True)


class BookStyleSerializer(serializers.Serializer):
    id = serializers.CharField(required=True, allow_blank=False, max_length=255)
    name = serializers.CharField(required=True, allow_blank=False, max_length=255)

    def create(self, validated_data):
        return BookStyle.objects.create(**validated_data)

    def update(self, instance, validated_data):
        instance.name = validated_data.get('name', instance.name)
        instance.save()
        return instance
