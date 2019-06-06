import unittest

from django.contrib.auth.models import User
from django.test import Client
from django.urls import reverse
import ommr4all.settings as settings
from rest_framework import status
from rest_framework.test import APITestCase
import time
import os
import sys
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', stream=sys.stdout)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Change database to test storage
settings.PRIVATE_MEDIA_ROOT = os.path.join(BASE_DIR, 'tests', 'storage')


class GenericTests(APITestCase):
    def test_ping(self):
        client = Client()
        response = client.get('/api/ping')
        self.assertEqual(response.status_code, 200)


class PermissionTests(APITestCase):

    def test_books(self):
        response = self.client.get('/api/books', format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)

    def test_get_preview_page(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/page_progress', format='json')
        self.assertEqual(response.status_code, status.HTTP_401_UNAUTHORIZED, response.content)


class OperationTests(APITestCase):
    def setUp(self):
        self.username = 'user'
        self.password = 'user'
        self.data = {
            'username': self.username,
            'password': self.password
        }
        # URL using path name
        url = reverse('jwtAuth')

        # Since Force Authentication method doesen't work, we have to create a user as
        # a workaround in order to authentication works.
        user = User.objects.create_superuser(username='user', email='user@mail.com', password='user')
        self.assertEqual(user.is_active, 1, 'Active User')

        # First post to get the JWT token
        response = self.client.post(url, self.data, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        token = response.data['token']
        # Next post/get's will require the token to connect
        self.client.credentials(HTTP_AUTHORIZATION='JWT {0}'.format(token))

    def test_books(self):
        response = self.client.get('/api/books', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertEqual('demo', response.data['books'][0]['id'], response.content)

    def test_pcgts_content(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/pcgts', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def test_statistics_content(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/statistics', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def test_page_progress_content(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/page_progress', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def test_save_page_without_lock(self):
        page = 'page_test_lock'
        response = self.save_page(page, {})
        self.assertEqual(response.status_code, status.HTTP_423_LOCKED, response.content)
        self.lock_page(page)
        response = self.save_page(page, {})
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.unlock_page(page)

    def test_get_preview_page(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/page_progress', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def _test_line_detection_of_page(self, page, n_lines):
        self.lock_page(page)
        response = self.client.put('/api/book/demo/page/{}/operation/staffs'.format(page), '{}', format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/demo/page/{}/operation/staffs/task/{}'.format(page, taskid)
        response = self.client.post(url, '{}', format='json')
        data = response.data
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
        self.assertEqual(len(data['staffs']), n_lines)
        self.unlock_page(page)

    def test_line_detection_001(self):
        self._test_line_detection_of_page('page_test_staff_line_detection_001', 9)

    def test_line_detection_002(self):
        self._test_line_detection_of_page('page_test_staff_line_detection_002', 9)

    def _test_layout_detection_of_page(self, page):
        self.lock_page(page)

        response = self.client.put('/api/book/demo/page/{}/operation/layout'.format(page), '{}', format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/demo/page/{}/operation/layout/task/{}'.format(page, taskid)
        response = self.client.post(url, '{}', format='json')
        data = response.data
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
        self.unlock_page(page)

        # Todo check data

    def test_layout_detection_001(self):
        self._test_layout_detection_of_page('page_test_layout_detection_001')

    def test_layout_detection_002(self):
        self._test_layout_detection_of_page('page_test_layout_detection_002')

    def _test_symbol_detection_of_page(self, page):
        self.lock_page(page)
        response = self.client.put('/api/book/demo/page/{}/operation/symbols'.format(page), '{}', format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/demo/page/{}/operation/symbols/task/{}'.format(page, taskid)
        response = self.client.post(url, '{}', format='json')
        data = response.data
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
        self.unlock_page(page)

    def test_symbol_detection_001(self):
        self._test_symbol_detection_of_page('page_test_symbol_detection_001')

    def test_symbol_detection_002(self):
        self._test_symbol_detection_of_page('page_test_symbol_detection_002')

    def save_page(self, page, json):
        return self.client.post('/api/book/demo/page/{}/operation/save'.format(page), json, format='json')

    def lock_page(self, page):
        payload = {}
        response = self.client.put('/api/book/demo/page/{}/lock'.format(page), payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def unlock_page(self, page):
        payload = {}
        response = self.client.delete('/api/book/demo/page/{}/lock'.format(page), payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)


if __name__ == '__main__':
    unittest.main()
