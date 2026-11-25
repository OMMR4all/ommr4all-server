import os
import shutil

from django.core.management.base import BaseCommand

import ommr4all.settings as settings


class Command(BaseCommand):
    help = 'Copy the default models shipped with the server into OMMR4ALL_DEFAULT_MODELS_ROOT, ' \
           'for every <style>/<model_dir> that is missing there. Existing directories are ' \
           'never touched, so a default model set by an administrator survives a restart. ' \
           'Does nothing when the variable is unset (target == shipped location).'

    def add_arguments(self, parser):
        parser.add_argument('--dry-run', action='store_true',
                            help='Only print what would be copied')

    def handle(self, *args, **options):
        # read through the module, not imported by value: the settings are patched in tests
        source = os.path.realpath(settings.SHIPPED_DEFAULT_MODELS_ROOT)
        target = os.path.realpath(settings.DEFAULT_MODELS_ROOT)

        if source == target:
            self.stdout.write('Default models are used in place ({}), nothing to seed'.format(target))
            return

        if not os.path.isdir(source):
            self.stderr.write(self.style.WARNING('No shipped default models at {}'.format(source)))
            return

        copied, skipped, failed = 0, 0, 0
        for style in sorted(_list_dirs(source)):
            for model_dir in sorted(_list_dirs(os.path.join(source, style))):
                src = os.path.join(source, style, model_dir)
                dst = os.path.join(target, style, model_dir)
                if os.path.exists(dst):
                    skipped += 1
                    continue
                if options['dry_run']:
                    self.stdout.write('would copy {} -> {}'.format(src, dst))
                    copied += 1
                    continue
                try:
                    os.makedirs(os.path.join(target, style), exist_ok=True)
                    # copy to a temporary name first: a crash mid-copy would otherwise leave a
                    # half-written model directory behind that this command then skips forever
                    tmp = dst + '.seeding'
                    shutil.rmtree(tmp, ignore_errors=True)
                    shutil.copytree(src, tmp)
                    os.replace(tmp, dst)
                    copied += 1
                    self.stdout.write('seeded {}/{}'.format(style, model_dir))
                except Exception as e:
                    failed += 1
                    self.stderr.write(self.style.ERROR(
                        'could not seed {}/{}: {}'.format(style, model_dir, e)))

        self.stdout.write('Default models in {}: {} seeded, {} already present{}'.format(
            target, copied, skipped, ', {} failed'.format(failed) if failed else ''))
        if failed:
            # a missing default model makes predictions fail later, so do not pretend success
            raise SystemExit(1)


def _list_dirs(path) -> list:
    try:
        return [e.name for e in os.scandir(path) if e.is_dir()]
    except OSError:
        return []
