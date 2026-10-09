import numpy as np
import pandas as pd
import pytest
from dalaflood.level import build_frame,make_windows,archived_weather,LevelLSTM,fit_rating_curve

def data():
    dates=pd.date_range('2024-01-01',periods=90)
    q=pd.Series(10.,index=dates)
    met=pd.DataFrame({'temperature_2m_mean':1.,'precipitation_sum':2.,'snow_depth_max':0.,'soil_temperature_7_to_28cm_mean':1.},index=dates)
    return q,met

def test_causal_fill_and_original_targets():
    q,met=data();q.iloc[35]=np.nan
    before=build_frame(q,met)
    q.iloc[36:]=1000;after=build_frame(q,met)
    pd.testing.assert_frame_equal(before.iloc[:36],after.iloc[:36])
    assert np.isnan(before['target_logq'].iloc[35])
    _,_,_,I=make_windows(before,noise=False)
    assert not any(i<35<=i+3 for i in I)

def test_leading_gap_not_filled_from_future():
    q,met=data();q.iloc[:5]=np.nan
    assert build_frame(q,met)['logq'].iloc[:5].isna().all()

def test_archive_rejects_unavailable_forecasts():
    issue=pd.Timestamp('2024-02-10')
    f=pd.DataFrame([dict(issue_time=issue,issued_at=issue-pd.Timedelta(days=1),valid_time=issue+pd.Timedelta(days=h),temperature_2m_mean=10.,precipitation_sum=2.) for h in range(1,4)])
    assert archived_weather([issue],f).shape==(1,3,2)
    f['issued_at']=issue+pd.Timedelta(minutes=1)
    with pytest.raises(ValueError):archived_weather([issue],f)

def test_invalid_windows():
    q,met=data()
    with pytest.raises(ValueError):make_windows(build_frame(q,met),np.array([-1,1,90]))

def test_conformal_never_inverts_interval():
    m=LevelLSTM.__new__(LevelLSTM)
    m._raw=lambda h,f: np.tile([0.,1.,2.],(20,3,1))
    margin=m.conformalize(None,None,np.ones((20,3)))
    assert (margin>=0).all()
    with pytest.raises(ValueError):m.conformalize(None,None,np.ones((20,3)),coverage=1)

def test_insufficient_rating_curve():
    with pytest.raises(ValueError):fit_rating_curve([1,1,1],[2,2,2])
