from datetime import datetime

from google.protobuf.timestamp_pb2 import Timestamp

from ta2.ta3.core_servicer import camel_case, dt2ts


def test_camel_case():
    assert camel_case('test') == 'test'
    assert camel_case('this is a test') == 'this is a test'
    assert camel_case('this_is_a_test') == 'thisIsATest'
    assert camel_case('THIS_IS_A_TEST') == 'THISISATEST'


def test_dt2ts():
    dt = datetime.strptime('21/11/06 16:30', '%d/%m/%y %H:%M')

    assert dt2ts(None) is None
    assert dt2ts(dt) == Timestamp(seconds=1164126600)
