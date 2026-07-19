from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from database.models.book_index import PageEditLock


class Command(BaseCommand):
    help = 'Release page edit locks older than the TTL (abandoned sessions). ' \
           'Locks are also auto-expired lazily on access; this command cleans up in bulk.'

    def add_arguments(self, parser):
        parser.add_argument('--ttl', type=float, default=None,
                            help='Age in hours after which a lock counts as stale '
                                 '(default: settings.PAGE_EDIT_LOCK_TTL_HOURS, {})'.format(
                                     getattr(settings, 'PAGE_EDIT_LOCK_TTL_HOURS', 12)))
        parser.add_argument('--dry-run', action='store_true',
                            help='Only print what would be released')

    def handle(self, *args, **options):
        ttl_hours = options['ttl']
        if ttl_hours is None:
            ttl_hours = getattr(settings, 'PAGE_EDIT_LOCK_TTL_HOURS', 12)
        if not ttl_hours:
            self.stdout.write('Lock expiry is disabled (TTL is 0), nothing to do')
            return

        cutoff = timezone.now() - timedelta(hours=ttl_hours)
        stale = PageEditLock.objects.filter(acquired_at__lt=cutoff) \
            .select_related('page', 'page__book', 'user')
        n = 0
        for lock in stale:
            self.stdout.write('{}/{}: lock of {} acquired {}'.format(
                lock.page.book.name, lock.page.name, lock.user.username, lock.acquired_at.isoformat()))
            n += 1
        if not options['dry_run']:
            stale.delete()
        self.stdout.write(self.style.SUCCESS(
            '{} {} stale lock(s) (TTL {}h)'.format('would release' if options['dry_run'] else 'released', n, ttl_hours)))
