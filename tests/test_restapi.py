import unittest

from django.contrib.auth.models import User
from django.test import Client
from django.urls import reverse
import ommr4all.settings as settings
from rest_framework import status
from rest_framework.test import APITestCase
import time
import os

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
        self.create_data()

    def test_books(self):
        response = self.client.get('/api/books', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
        self.assertEqual('demo', response.data['books'][0]['id'], response.content)

    def create_data(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/pcgts', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def test_get_preview_page(self):
        response = self.client.get('/api/book/demo/page/page00000001/content/page_progress', format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def test_line_detection(self):
        response = self.client.put('/api/book/demo/page/page00000001/operation/staffs', '{}', format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/demo/page/page00000001/operation/staffs/task/' + taskid
        self.lock_page()
        response = self.client.post(url, '{}', format='json')
        data = response.data
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
        self.assertEqual(len(data['staffs']), 9)

    def test_layout_detection(self):
        self.lock_page()
        self.save_page()
        response = self.client.put('/api/book/demo/page/page00000001/operation/layout', '{}', format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/demo/page/page00000001/operation/layout/task/' + taskid
        self.lock_page()
        response = self.client.post(url, '{}', format='json')
        data = response.data
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data
        # Todo check data

    def test_symbol_detection(self):
        response = self.client.put('/api/book/demo/page/page00000001/operation/symbols', '{}', format='json')
        self.assertEqual(response.status_code, status.HTTP_202_ACCEPTED, response.content)
        taskid = response.data['task_id']
        url = '/api/book/demo/page/page00000001/operation/symbols/task/' + taskid
        self.lock_page()
        response = self.client.post(url, '{}', format='json')
        data = response.data
        while data['status']['code'] == 1 or data['status']['code'] == 0:
            time.sleep(1)
            response = self.client.post(url, '{}', format='json')
            self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)
            data = response.data

    def save_page(self):
        import json
        payload = '{"meta":{"creator":"","created":"2019-05-23 10:37:04.312693","lastChange":"2019-05-23 10:37:04.312701"},"page":{"textRegions":[],"musicRegions":[{"id":"page_0:block_0","coords":"","musicLines":[{"id":"page_0:block_0:line_0","coords":"","staffLines":[{"id":"page_0:block_0:line_0:sl_0","coords":"114,159 172,150 351.5,144.5 460,144 556,141.5 727.5,141.5 778,143 781,144","highlighted":false,"space":false},{"id":"page_0:block_0:line_0:sl_1","coords":"114,154.5 346,166.5 486,165 563,163 778,162.5 781,163.5","highlighted":false,"space":false},{"id":"page_0:block_0:line_0:sl_2","coords":"114,179.5 170,180.5 197,179 313,187 367,184 405,185 556,182.5 703,182 778,183.5 782,178","highlighted":false,"space":false},{"id":"page_0:block_0:line_0:sl_3","coords":"114,214.5 160,215.5 340,205 599,201.5 702,201 781,202.5","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_1","coords":"","musicLines":[{"id":"page_0:block_1:line_1","coords":"","staffLines":[{"id":"page_0:block_1:line_1:sl_4","coords":"120,271 183,271 321,262 477,260.5 517,259 618,260 759,259.5 846,263","highlighted":false,"space":false},{"id":"page_0:block_1:line_1:sl_5","coords":"118,293 214,289 388.5,283.5 435,283.5 559,280.5 716.5,282 768.5,280.5 849,282","highlighted":false,"space":false},{"id":"page_0:block_1:line_1:sl_6","coords":"118,314 226,309.5 301,308.5 330,307 392,306.5 487,304 616.5,302.5 693,304.5 744,305 797,304.5 849,305.5","highlighted":false,"space":false},{"id":"page_0:block_1:line_1:sl_7","coords":"118,333.5 202,331 252,331 363,329 603.5,326.5 656,328 716.5,327 776,327 820.5,328 848,327.5","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_2","coords":"","musicLines":[{"id":"page_0:block_2:line_2","coords":"","staffLines":[{"id":"page_0:block_2:line_2:sl_8","coords":"91,385.5 139,383.5 168,384 216,383.5 321,381 552.5,378.5 685,379 771,381.5","highlighted":false,"space":false},{"id":"page_0:block_2:line_2:sl_9","coords":"91,407 208,405.5 278,403 539,399.5 717.5,400 771,401.5","highlighted":false,"space":false},{"id":"page_0:block_2:line_2:sl_10","coords":"91,429 214,427.5 366,424.5 599,423 739,423 771,424","highlighted":false,"space":false},{"id":"page_0:block_2:line_2:sl_11","coords":"91,450.5 126,448.5 514.5,444.5 772,446.5","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_3","coords":"","musicLines":[{"id":"page_0:block_3:line_3","coords":"","staffLines":[{"id":"page_0:block_3:line_3:sl_12","coords":"87,505 113.5,504 353,501.5 509,501.5 545,500.5 683,501 716,500.5 753,502","highlighted":false,"space":false},{"id":"page_0:block_3:line_3:sl_13","coords":"87,528.5 210,528.5 324,527 432,527.5 611,526 642.5,527.5 672,527.5 753,526","highlighted":false,"space":false},{"id":"page_0:block_3:line_3:sl_14","coords":"87,550.5 133,549.5 172,550.5 255,550 466,550.5 572,550 615,549 652,550 731,550.5 754,550","highlighted":false,"space":false},{"id":"page_0:block_3:line_3:sl_15","coords":"91,570 159,570 196,571 232,570.5 432.5,571 511,569.5 619,569 753,571","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_4","coords":"","musicLines":[{"id":"page_0:block_4:line_4","coords":"","staffLines":[{"id":"page_0:block_4:line_4:sl_16","coords":"92,622.5 191,623 384.5,622 419,623 529,621.5 775,622","highlighted":false,"space":false},{"id":"page_0:block_4:line_4:sl_17","coords":"90,647 156,647 189,648 288,646.5 329,647.5 381,647 421,648.5 589,646 774,646.5","highlighted":false,"space":false},{"id":"page_0:block_4:line_4:sl_18","coords":"93,667.5 243,668 356,667 444,668 606,666.5 706,668.5 750,667 774,668","highlighted":false,"space":false},{"id":"page_0:block_4:line_4:sl_19","coords":"90,688 134,688 177,689.5 220.5,690 336,689.5 396,690 428,691.5 512,691.5 538,690.5 566,691.5 600.5,691 669.5,693 773,693","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_5","coords":"","musicLines":[{"id":"page_0:block_5:line_5","coords":"","staffLines":[{"id":"page_0:block_5:line_5:sl_20","coords":"91,749.5 143,750 167,751.5 660.5,753.5 708,752.5 768,752.5","highlighted":false,"space":false},{"id":"page_0:block_5:line_5:sl_21","coords":"89,772 135,771.5 225,773 296,772.5 337,771 467,773 511.5,772.5 677,774 717,773 769,773.5","highlighted":false,"space":false},{"id":"page_0:block_5:line_5:sl_22","coords":"89,794 568.5,793.5 717,794.5 768,793.5","highlighted":false,"space":false},{"id":"page_0:block_5:line_5:sl_23","coords":"89,813 126.5,812.5 164,813.5 331,813.5 361.5,812.5 652,815 740,815 768,821.5","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_6","coords":"","musicLines":[{"id":"page_0:block_6:line_6","coords":"","staffLines":[{"id":"page_0:block_6:line_6:sl_24","coords":"82,862.5 312,863 353,862 393,863.5 540,863 676,865.5 715,865.5 772,867","highlighted":false,"space":false},{"id":"page_0:block_6:line_6:sl_25","coords":"82,887.5 232,887 294,886 539,886.5 590,886 772,888.5","highlighted":false,"space":false},{"id":"page_0:block_6:line_6:sl_26","coords":"82,908 184,909 353,909 399,908 442,910 481.5,909.5 578,910 773,913","highlighted":false,"space":false},{"id":"page_0:block_6:line_6:sl_27","coords":"82,929.5 145.5,929.5 206,931 298,930.5 389,931 423,932 563.5,932 766,936.5 768,933.5","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_7","coords":"","musicLines":[{"id":"page_0:block_7:line_7","coords":"","staffLines":[{"id":"page_0:block_7:line_7:sl_28","coords":"77,983 113,985 139,984 187.5,986 316,984.5 433,985.5 465,984.5 669,985.5 765,987 770,991","highlighted":false,"space":false},{"id":"page_0:block_7:line_7:sl_29","coords":"77,1007.5 231,1008 333,1006 369,1006 401,1007 550,1005 631,1006 682,1005.5 765,1007 770,1008","highlighted":false,"space":false},{"id":"page_0:block_7:line_7:sl_30","coords":"77,1030.5 113,1030 195,1030.5 293.5,1028.5 410.5,1029 584,1027 737,1028 771,1030","highlighted":false,"space":false},{"id":"page_0:block_7:line_7:sl_31","coords":"77,1052 133,1052 234.5,1054 312,1053 383,1054 483.5,1054 556,1052.5 769,1054 771,1052","highlighted":false,"space":false}],"symbols":[]}]},{"id":"page_0:block_8","coords":"","musicLines":[{"id":"page_0:block_8:line_8","coords":"","staffLines":[{"id":"page_0:block_8:line_8:sl_32","coords":"77,1101.5 165,1102 245,1103.5 328,1102 473,1104.5 610.5,1104.5 762,1106.5","highlighted":false,"space":false},{"id":"page_0:block_8:line_8:sl_33","coords":"77,1124.5 126.5,1123.5 197,1126 253.5,1125 376,1124.5 412,1126 491.5,1127.5 607,1128 759,1130 763,1131","highlighted":false,"space":false},{"id":"page_0:block_8:line_8:sl_34","coords":"82,1145.5 143,1145.5 206,1147.5 245,1146.5 355,1146 419,1147.5 562,1149 759,1153 762,1154","highlighted":false,"space":false},{"id":"page_0:block_8:line_8:sl_35","coords":"77,1168 222.5,1170 374,1169.5 440.5,1172 543,1172.5 648.5,1175 759,1175.5 763,1178.5","highlighted":false,"space":false}],"symbols":[]}]}],"imageFilename":"color_deskewed.jpg","imageWidth":1024,"imageHeight":1532,"readingOrder":{"lyricsReadingOrder":[]},"annotations":{"connections":[]},"comments":{"comments":[]}}}'
        payload = json.loads(payload)
        response = self.client.post('/api/book/demo/page/page00000001/operation/save', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)

    def lock_page(self):
        payload = {}
        response = self.client.put('/api/book/demo/page/page00000001/lock', payload, format='json')
        self.assertEqual(response.status_code, status.HTTP_200_OK, response.content)


if __name__ == '__main__':
    unittest.main()
