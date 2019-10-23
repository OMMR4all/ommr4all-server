import unittest
import django
django.setup()

from django.contrib.auth.models import User
from django.http import FileResponse
from django.test import Client
from django.urls import reverse
import ommr4all.settings as settings
from rest_framework import status
from rest_framework.test import APITestCase
import time
import os
import sys
import logging
import json
from database.database_page import DatabaseBook, DatabasePage
from shared.jsonparsing import drop_all_attributes
from restapi.views.bookoperations import AlgorithmRequest, AlgorithmPredictorParams, AlgorithmTypes
from restapi.operationworker.taskrunners.pageselection import PageCount, PageSelectionParams

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
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        body = json.loads(response.content)
        self.assertListEqual(body['books'], [])
        self.assertEqual(body['totalPages'], 0)

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
        self.assertEqual(response.status_code, status.HTTP_200_OK, response)

    def test_save_page_without_lock(self):
        page = 'page_test_lock'
        response = self.save_page(page, {})
        self.assertEqual(response.status_code, status.HTTP_423_LOCKED, response)
        self.lock_page(page)
        response = self.save_page(page, {})
        self.assertEqual(response.status_code, status.HTTP_200_OK, response)
        self.unlock_page(page)

    def test_save_statistics_without_lock(self):
        page = 'page_test_lock'

        def save_statistics():
            return self.client.put('/api/book/demo/page/{}/content/statistics'.format(page), {}, format='json')

        response = save_statistics()
        self.assertEqual(response.status_code, status.HTTP_423_LOCKED, response)
        self.lock_page(page)
        response = save_statistics()
        self.assertEqual(response.status_code, status.HTTP_200_OK, response)
        self.unlock_page(page)

    def test_save_page_progress_without_lock(self):
        page = 'page_test_lock'

        def save():
            return self.client.put('/api/book/demo/page/{}/content/page_progress'.format(page), {}, format='json')

        response = save()
        self.assertEqual(response.status_code, status.HTTP_423_LOCKED, response)
        self.lock_page(page)
        response = save()
        self.assertEqual(response.status_code, status.HTTP_200_OK, response)
        self.unlock_page(page)

    def test_get_preview_page(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/page_progress', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response)

    def _test_preprocessing_of_page(self, page: DatabasePage):
        self.call_operation(page, AlgorithmTypes.PREPROCESSING.value, {'automaticLd': True})

    def test_preprocessing_001(self):
        page = DatabaseBook('demo').page('page_test_preprocessing_001')
        self._test_preprocessing_of_page(page)

    def _test_line_detection_of_page(self, page: str, n_lines, algorithm: AlgorithmTypes):
        data = self._test_predictor(page, algorithm)
        self.assertEqual(len(data['staffs']), n_lines)

    def test_line_detection_001(self):
        self._test_line_detection_of_page('page_test_staff_line_detection_001', 9, AlgorithmTypes.STAFF_LINES_PC)

    def test_line_detection_002(self):
        self._test_line_detection_of_page('page_test_staff_line_detection_002', 9, AlgorithmTypes.STAFF_LINES_PC)

    def test_layout_detection_complex_standard_001(self):
        self._test_predictor('page_test_layout_detection_001', AlgorithmTypes.LAYOUT_COMPLEX_STANDARD)

    def test_layout_detection_simple_bounding_boxes_001(self):
        self._test_predictor('page_test_layout_detection_001', AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES)

    def test_layout_detection_simple_lyrics_001(self):
        self._test_predictor('page_test_layout_detection_001', AlgorithmTypes.LAYOUT_SIMPLE_LYRICS)

    def test_layout_detection_complex_standard_002(self):
        self._test_predictor('page_test_layout_detection_002', AlgorithmTypes.LAYOUT_COMPLEX_STANDARD)

    def test_layout_detection_simple_bounding_boxes_002(self):
        self._test_predictor('page_test_layout_detection_002', AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES)

    def test_layout_detection_simple_lyrics_002(self):
        self._test_predictor('page_test_layout_detection_002', AlgorithmTypes.LAYOUT_SIMPLE_LYRICS)

    def test_symbol_detection_001(self):
        self._test_predictor('page_test_symbol_detection_001', AlgorithmTypes.SYMBOLS_PC)

    def test_symbol_detection_002(self):
        self._test_predictor('page_test_symbol_detection_002', AlgorithmTypes.SYMBOLS_PC)

    def test_text_recognition_001(self):
        self._test_predictor('page_test_text_recognition_001', AlgorithmTypes.OCR_CALAMARI)

    def test_syllable_detection_from_text_001(self):
        self._test_predictor('page_test_syllable_detection_001', AlgorithmTypes.SYLLABLES_FROM_TEXT)

    def test_syllable_detection_in_order_001(self):
        self._test_predictor('page_test_syllable_detection_001', AlgorithmTypes.SYLLABLES_IN_ORDER)

    def test_export_monodi(self):
        page = DatabaseBook('demo').page('page_test_monodi_export_001')
        response: FileResponse = self.client.post('/api/book/{}/download/monodiplus.json'.format(page.book.book),
                                                  {'pages': [page.page]}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.filename)
        data = b''.join(response.streaming_content).decode('utf-8')
        data = json.loads(data)
        drop_all_attributes(data, 'uuid')
        with open(page.local_file_path('monodi.json')) as f:
            self.assertEqual(data, json.load(f))

    def test_export_annotations(self):
        page = DatabaseBook('demo').page('page_test_monodi_export_001')
        self.call_operation(page, 'preprocessing', {'automaticLd': True})
        self.client.post('/api/book/{}/page/{}/operation')
        response: FileResponse = self.client.post('/api/book/{}/download/annotations.zip'.format(page.book.book),
                                                  {'pages': [page.page]}, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.filename)
        import io
        import zipfile
        by = io.BytesIO(b''.join(response.streaming_content))
        files_to_expect = ['color_original', 'color_norm_x2', 'binary_norm_x2', 'pcgts', 'meta']
        with zipfile.ZipFile(by) as f:
            zip_files = [z.filename for z in f.infolist()]
            files_to_expect = ["{}/{}{}".format(file, page.page, page.file(file).ext()) for file in files_to_expect]
            self.assertListEqual(files_to_expect, zip_files)
            self.assertEqual(len(zip_files), len(files_to_expect))

    def _test_list_models(self, operation: AlgorithmTypes):
        from database.database_available_models import DatabaseAvailableModels
        from database.model import ModelsId, MetaId
        book = DatabaseBook('demo')
        response = self.client.get('/api/book/{}/operation/{}/models'.format(book.book, operation.value))
        self.assertEqual(response.status_code, status.HTTP_200_OK, response)

        models_id = ModelsId.from_internal('french14', AlgorithmTypes.STAFF_LINES_PC)
        models = DatabaseAvailableModels.from_dict(response.data)
        self.assertEqual(models.newest_model, None)
        self.assertEqual(models.selected_model.id, str(MetaId(models_id, models_id.algorithm_type.value)))
        self.assertEqual(models.default_book_style_model.id, str(MetaId(models_id, models_id.algorithm_type.value)))
        self.assertListEqual(models.book_models, [])
        self.assertListEqual(models.models_of_same_book_style, [])

    def test_list_models(self):
        self._test_list_models(AlgorithmTypes.STAFF_LINES_PC)

    def save_page(self, page, data):
        return self.client.put('/api/book/demo/page/{}/content/pcgts'.format(page), data, format='json')

    def lock_page(self, page):
        payload = {}
        response = self.client.put('/api/book/demo/page/{}/lock'.format(page), payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def unlock_page(self, page):
        payload = {}
        response = self.client.delete('/api/book/demo/page/{}/lock'.format(page), payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def _test_predictor(self, page, algorithm: AlgorithmTypes):
        page = DatabaseBook('demo').page(page)
        data = self.call_operation(page, algorithm.value, {
            'pcgts': page.pcgts().to_json(),
        })
        return data

    def call_operation(self, page: DatabasePage, operation: str, data=None):
        data = data if data else {}
        self.lock_page(page.page)
        response = self.client.put('/api/book/{}/page/{}/operation/{}/'.format(page.book.book, page.page, operation), data, format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/{}/page/{}/operation/{}/task/{}'.format(page.book.book, page.page, operation, taskid)
        response = self.client.post(url, '{}', format='json')
        data = response.data
        timeout = 0
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
            timeout += 1
            if timeout > 10:
                self.failureException("Timeout ({}s) when calling book operation at {}".format(timeout, url))
        self.unlock_page(page.page)
        return data

    # Test Book Operations
    # ========================================

    def test_book_preprocessing(self):
        book = DatabaseBook('demo')
        params = AlgorithmRequest(
            store_to_pcgts=False,
            params=AlgorithmPredictorParams(
                automaticLd=True
            ),
            selection=PageSelectionParams(
                count=PageCount.CUSTOM,
                pages=['page_test_preprocessing_001'],
            )
        )
        self.call_book_operation(book, AlgorithmTypes.PREPROCESSING, params)

    def test_book_line_detection(self):
        book = DatabaseBook('demo')
        params = AlgorithmRequest(
            store_to_pcgts=False,
            params=AlgorithmPredictorParams(),
            selection=PageSelectionParams(
                count=PageCount.CUSTOM,
                pages=['page_test_staff_line_detection_001', 'page_test_staff_line_detection_002'],
            )
        )
        self.call_book_operation(book, AlgorithmTypes.STAFF_LINES_PC, params)

    def test_book_layout_detection_complex(self):
        book = DatabaseBook('demo')
        params = AlgorithmRequest(
            store_to_pcgts=False,
            params=AlgorithmPredictorParams(
            ),
            selection=PageSelectionParams(
                count=PageCount.CUSTOM,
                pages=['page_test_layout_detection_001', 'page_test_layout_detection_002'],
            )
        )
        self.call_book_operation(book, AlgorithmTypes.LAYOUT_COMPLEX_STANDARD, params)

    def test_book_layout_detection_simple(self):
        book = DatabaseBook('demo')
        params = AlgorithmRequest(
            store_to_pcgts=False,
            params=AlgorithmPredictorParams(
            ),
            selection=PageSelectionParams(
                count=PageCount.CUSTOM,
                pages=['page_test_layout_detection_001', 'page_test_layout_detection_002'],
            )
        )
        self.call_book_operation(book, AlgorithmTypes.LAYOUT_SIMPLE_BOUNDING_BOXES, params)

    def test_book_symbol_detection(self):
        book = DatabaseBook('demo')
        params = AlgorithmRequest(
            store_to_pcgts=False,
            params=AlgorithmPredictorParams(
            ),
            selection=PageSelectionParams(
                count=PageCount.CUSTOM,
                pages=['page_test_symbol_detection_001', 'page_test_symbol_detection_002'],
            )
        )
        self.call_book_operation(book, AlgorithmTypes.SYMBOLS_PC, params)

    def call_book_operation(self, book: DatabaseBook, operation: AlgorithmTypes, data: AlgorithmRequest = None):
        data = data if data else AlgorithmRequest()
        response = self.client.put('/api/book/{}/operation/{}/'.format(book.book, operation.value), data.to_dict(), format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response)
        taskid = response.data['task_id']
        url = '/api/book/{}/operation/{}/task/{}'.format(book.book, operation.value, taskid)
        response = self.client.post(url, '{}', format='json')
        data = response.data
        timeout = 0
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
            timeout += 1
            if timeout > 10:
                self.failureException("Timeout ({}s) when calling book operation at {}".format(timeout, url))

        return data


if __name__ == '__main__':
    unittest.main()
