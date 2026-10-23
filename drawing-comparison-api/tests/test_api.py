import io
from pathlib import Path
import cv2
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient
from api import app, slot
from comparison import compare, rectify, overlay_camera

client = TestClient(app)
samples = Path(__file__).resolve().parents[1] / 'samples'

def request(drawing='drawing_identical.png', **kwargs):
    return client.post('/compare', files={'original_image': ('original.png', (samples/'reference.png').read_bytes(), 'image/png'), 'drawing_image': ('drawing.png', (samples/drawing).read_bytes(), 'image/png')}, **kwargs)

@pytest.mark.parametrize('drawing', ['drawing_identical.png','drawing_missing_window.png','drawing_shifted_8mm.png'])
def test_exact_client_algorithm_parity(drawing):
    response=request(drawing)
    assert response.status_code == 200
    camera=np.array(Image.open(samples/drawing).convert('RGB'))
    reference=np.array(Image.open(samples/'reference.png').convert('RGB'))
    h,w=camera.shape[:2]
    corrected,H=rectify(camera,[[0,0],[w-1,0],[w-1,h-1],[0,h-1]],300,400)
    reference=cv2.resize(reference,(corrected.shape[1],corrected.shape[0]),interpolation=cv2.INTER_AREA)
    expected=overlay_camera(camera,compare(reference,corrected,300,400),H)
    actual=np.array(Image.open(io.BytesIO(response.content)))
    assert actual.shape == camera.shape
    assert np.array_equal(actual,expected)
    assert response.headers['content-type']=='image/png'
    if drawing=='drawing_identical.png':
        assert float(response.headers['x-comparison-score']) > .99
        assert ((actual[:,:,1]>actual[:,:,0]+30) & (actual[:,:,1]>actual[:,:,2]+30)).any()
    if drawing=='drawing_shifted_8mm.png':
        assert ((actual[:,:,0].astype(int)>actual[:,:,1].astype(int)+60)).any()

def test_errors_and_health(monkeypatch):
    assert client.get('/health').json()=={'status':'ok'}
    assert client.get('/').status_code==200
    assert client.post('/compare').status_code==422
    assert request(data={'tolerance_mm':'nan'}).status_code==422
    monkeypatch.setenv('ALCHROMA_API_KEY','test-secret')
    assert request().status_code==401
    assert request(headers={'X-API-Key':'test-secret'}).status_code==200

def test_busy():
    slot.acquire()
    try:
        assert request().status_code==503
        assert client.get('/health').status_code==200
    finally: slot.release()

def test_invalid_and_blank():
    sample=(samples/'reference.png').read_bytes()
    response=client.post('/compare',files={'original_image':('x.png',b'not an image'),'drawing_image':('x.png',sample)})
    assert response.status_code==415
    blank=io.BytesIO();Image.new('RGB',(100,100),'white').save(blank,format='PNG')
    response=client.post('/compare',files={'original_image':('x.png',blank.getvalue()),'drawing_image':('x.png',sample)})
    assert response.status_code==422
    response=client.post('/compare',files={'original_image':('x.png',b'x'*(10*1024*1024+1)),'drawing_image':('x.png',sample)})
    assert response.status_code==413

def test_exif_and_reference_size():
    picture=Image.open(samples/'drawing_identical.png').convert('RGB').resize((160,240))
    exif=Image.Exif();exif[274]=6
    jpeg=io.BytesIO();picture.save(jpeg,format='JPEG',exif=exif)
    response=client.post('/compare',files={'original_image':('original.png',(samples/'reference.png').read_bytes()),'drawing_image':('drawing.jpg',jpeg.getvalue())})
    assert response.status_code==200
    assert Image.open(io.BytesIO(response.content)).size==(240,160)


def test_canvas_fields_removed_and_legacy_values_cannot_change_output():
    page=client.get('/').text
    schema=client.get('/openapi.json').json()
    import json
    for field in ('canvas_width_mm', 'canvas_height_mm'):
        assert field not in page
        assert field not in json.dumps(schema)
    baseline=request()
    old_client=request(data={'canvas_width_mm':900, 'canvas_height_mm':200})
    assert baseline.status_code == old_client.status_code == 200
    assert baseline.content == old_client.content


def test_structured_errors():
    response=client.post('/compare')
    body=response.json()
    assert body['success'] is False
    assert body['error']['code']=='INVALID_INPUT'
    assert {item['field'] for item in body['error']['fields']} == {'body.original_image', 'body.drawing_image'}
    assert client.get('/missing').json()['error']['code']=='NOT_FOUND'
    assert client.get('/compare').json()['error']['code']=='METHOD_NOT_ALLOWED'
    slot.acquire()
    try:
        response=request()
        assert response.json()['error']['code']=='SERVER_BUSY'
        assert response.headers['retry-after']=='2'
    finally:
        slot.release()


def test_internal_failure_is_safe_and_recovers(monkeypatch):
    import api
    original=api.compare
    def broken(*args, **kwargs):
        raise RuntimeError('private implementation detail')
    monkeypatch.setattr(api, 'compare', broken)
    with TestClient(app, raise_server_exceptions=False) as safe_client:
        response=safe_client.post('/compare', files={
            'original_image':('ref.png',(samples/'reference.png').read_bytes()),
            'drawing_image':('draw.png',(samples/'drawing_identical.png').read_bytes())})
    assert response.status_code==500
    assert response.json()['error']['code']=='INTERNAL_ERROR'
    assert 'private implementation detail' not in response.text
    monkeypatch.setattr(api, 'compare', original)
    assert request().status_code==200
