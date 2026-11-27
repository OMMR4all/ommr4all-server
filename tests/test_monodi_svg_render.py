"""Rendering a monodi document to SVG (database/file_formats/exporter/monodi/svg_render.py).

The renderer is a JVM tool bundled in internal_storage. A machine without a Java runtime --
the Docker image was one -- made the launcher print "No java installations was detected" and
exit 1, which the views turned into `AttributeError: NoneType has no attribute 'group'`, i.e.
a 500 with a traceback and no hint at the cause.
"""
import logging
import os
import stat
import sys
import tempfile
import unittest
from unittest import mock

import ommr4all.settings as settings

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['OMMR4ALL_STORAGE_ROOT'] = os.path.join(BASE_DIR, 'tests', 'storage')
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')

import django
django.setup()

from django.contrib.auth.models import User
from django.test import TestCase as DjangoTestCase
from rest_framework.test import APIClient

from database.file_formats.exporter.monodi import svg_render
from restapi.models.error import ErrorCodes

SVG = '<svg xmlns="http://www.w3.org/2000/svg"><g class="note"/></svg>'


def _script(body: str) -> str:
    """A stand-in for bin/one-shot, so the tests need neither a JVM nor the real tool."""
    handle, path = tempfile.mkstemp(prefix='one-shot-', suffix='.sh')
    with os.fdopen(handle, 'w') as f:
        f.write('#!/bin/sh\n' + body)
    os.chmod(path, os.stat(path).st_mode | stat.S_IEXEC)
    return path


class TestRender(unittest.TestCase):
    def tearDown(self):
        for path in getattr(self, '_scripts', []):
            os.unlink(path)

    def _use(self, body: str):
        self._scripts = getattr(self, '_scripts', [])
        path = _script(body)
        self._scripts.append(path)
        return mock.patch.object(svg_render, 'SCRIPT_PATH', path)

    def test_the_svg_is_pulled_out_of_the_output(self):
        # the tool prints progress around the document, as the real one does
        with self._use('echo "rendering..."\necho \'{}\'\necho done\n'.format(SVG)):
            self.assertEqual(SVG, svg_render.render('{}', 500))

    def test_a_nonzero_exit_with_an_svg_still_counts(self):
        # deliberate: the launcher exits non-zero on recoverable complaints, which the
        # previous implementation also ignored
        with self._use('echo \'{}\'\nexit 3\n'.format(SVG)):
            self.assertEqual(SVG, svg_render.render('{}', 500))

    def test_a_missing_java_runtime_is_named(self):
        with self._use('echo\necho No java installations was detected.\nexit 1\n'):
            with self.assertRaises(svg_render.MonodiSvgRenderError) as caught:
                svg_render.render('{}', 500)
        self.assertIn('Java', str(caught.exception))

    def test_other_failures_report_the_output(self):
        with self._use('echo "boom: cannot parse the document" >&2\nexit 1\n'), \
                mock.patch.object(svg_render, 'java_available', return_value=True):
            with self.assertRaises(svg_render.MonodiSvgRenderError) as caught:
                svg_render.render('{}', 500)
        self.assertIn('no SVG', str(caught.exception))

    def test_a_missing_renderer_is_named(self):
        with mock.patch.object(svg_render, 'SCRIPT_PATH', '/nonexistent/one-shot'):
            with self.assertRaises(svg_render.MonodiSvgRenderError) as caught:
                svg_render.render('{}', 500)
        self.assertIn('missing', str(caught.exception))

    def test_a_hanging_renderer_is_cut_off(self):
        with self._use('sleep 5\n'), mock.patch.object(svg_render, 'RENDER_TIMEOUT_SECONDS', 1):
            with self.assertRaises(svg_render.MonodiSvgRenderError) as caught:
                svg_render.render('{}', 500)
        self.assertIn('did not finish', str(caught.exception))

    def test_render_file_writes_the_target(self):
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, 'out.svg')
            with self._use('shift\nwhile [ "$1" != "-o" ]; do shift; done\n'
                           'echo \'{}\' > "$2"\n'.format(SVG)):
                svg_render.render_file(os.path.join(d, 'in.json'), out)
            self.assertTrue(os.path.exists(out))

    def test_render_file_complains_when_nothing_was_written(self):
        with tempfile.TemporaryDirectory() as d:
            with self._use('exit 0\n'):
                with self.assertRaises(svg_render.MonodiSvgRenderError):
                    svg_render.render_file(os.path.join(d, 'in.json'),
                                           os.path.join(d, 'out.svg'))

    def test_the_shipped_renderer_is_where_it_is_expected(self):
        # it lives in internal_storage, which is relocatable; the path must follow
        self.assertTrue(svg_render.SCRIPT_PATH.startswith(settings.INTERNAL_STORAGE_ROOT))
        self.assertTrue(os.access(svg_render.SCRIPT_PATH, os.X_OK), svg_render.SCRIPT_PATH)


class TestSvgEndpoints(DjangoTestCase):
    """A failing renderer must answer with the API error contract, not a traceback."""

    def setUp(self):
        # a superuser, so that the book permissions of the test storage stay untouched
        # (resolve_user_permissions grants them full access without reading the file)
        self.user = User.objects.create_superuser('svg_admin', password='pw')
        self.client = APIClient()
        self.client.force_authenticate(user=self.user)

    def test_the_page_endpoint_reports_a_render_failure(self):
        with mock.patch.object(svg_render, 'render',
                               side_effect=svg_render.MonodiSvgRenderError('no Java runtime')):
            response = self.client.get('/api/book/demo/page/page00000001/svg/500')
        self.assertEqual(500, response.status_code, response.content)
        body = response.json()
        self.assertEqual(ErrorCodes.MONODI_SVG_RENDER_FAILED.value, body['errorCode'])
        self.assertIn('no Java runtime', body['developerMessage'])
        self.assertTrue(body['userMessage'])

    def test_the_page_endpoint_serves_the_svg(self):
        with mock.patch.object(svg_render, 'render', return_value=SVG):
            response = self.client.get('/api/book/demo/page/page00000001/svg/500')
        self.assertEqual(200, response.status_code, response.content)
        self.assertEqual('image/svg+xml', response['Content-Type'])
        self.assertEqual(SVG, response.content.decode())


if __name__ == '__main__':
    unittest.main()
