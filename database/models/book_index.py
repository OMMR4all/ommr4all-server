from django.conf import settings
from django.db import models


class BookIndex(models.Model):
    """Queryable mirror of a book folder under PRIVATE_MEDIA_ROOT.

    The folder stays the source of truth: every row is rebuildable from
    book_meta.json (see database/book_index.py) and carries the mtime it was
    derived from, so stale rows are detected by a cheap stat and self-heal.
    """
    name = models.CharField(max_length=255, primary_key=True)
    meta = models.JSONField(default=dict)
    meta_mtime = models.FloatField(default=0.0)

    # denormalized from `meta` for listing/sorting without JSON access
    display_name = models.CharField(max_length=255, blank=True, default='')
    notation_style = models.CharField(max_length=255, blank=True, default='')
    created = models.DateTimeField(null=True, blank=True)
    updated = models.DateTimeField(null=True, blank=True)
    updated_by = models.CharField(max_length=255, blank=True, default='')

    indexed_at = models.DateTimeField(auto_now=True)


class PageIndex(models.Model):
    """Mirror of a page folder: mtimes of its content files plus the progress
    fields needed for overview stats and training-page selection."""
    book = models.ForeignKey(BookIndex, on_delete=models.CASCADE, related_name='pages')
    name = models.CharField(max_length=255)

    pcgts_mtime = models.FloatField(default=0.0)      # 0.0 = file absent
    progress_mtime = models.FloatField(default=0.0)

    has_symbols = models.BooleanField(default=False)
    verified = models.BooleanField(default=False)
    # {'StaffLines': bool, 'Layout': bool, 'Symbols': bool, 'Text': bool}
    progress_locks = models.JSONField(default=dict)

    # lazily computed symbol/line counts for BookStatsView; null = not yet computed
    counts = models.JSONField(null=True, blank=True)
    counts_mtime = models.FloatField(default=0.0)

    class Meta:
        unique_together = [('book', 'name')]
        ordering = ['name']


class BookDocumentsIndex(models.Model):
    """Parse-avoiding mirror of book_documents.json (which stays the file of record)."""
    book = models.OneToOneField(BookIndex, on_delete=models.CASCADE, related_name='documents_index')
    documents = models.JSONField(null=True)
    file_mtime = models.FloatField(default=0.0)


class PageEditLock(models.Model):
    """Exclusive edit lock of a page. Ephemeral (not part of the book folder):
    a lock does not survive moving a book to another server."""
    page = models.OneToOneField(PageIndex, on_delete=models.CASCADE, related_name='edit_lock')
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    acquired_at = models.DateTimeField(auto_now_add=True)
