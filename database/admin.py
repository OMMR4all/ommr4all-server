from django.contrib import admin
from .models.bookstyles import BookStyle


class BookStyleAdmin(admin.ModelAdmin):
    pass


admin.site.register(BookStyle, BookStyleAdmin)
