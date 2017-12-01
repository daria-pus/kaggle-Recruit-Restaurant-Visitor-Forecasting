import numpy as np
import pandas as pd 

# originally from this kernel
# https://www.kaggle.com/guoqiangliang/median-by-dow-lb-0-517
test_df = pd.read_csv('sample_submission.csv')
test_df['store_id'], test_df['visit_date'] = test_df['id'].str[:20], test_df['id'].str[21:]
test_df.drop(['visitors'], axis=1, inplace=True)
test_df['visit_date'] = pd.to_datetime(test_df['visit_date'])
#%%
air_data = pd.read_csv('air_visit_data.csv', parse_dates=['visit_date'])
air_data['dow'] = air_data['visit_date'].dt.dayofweek
train = air_data[air_data['visit_date'] > '2017-01-28'].reset_index()
train['dow'] = train['visit_date'].dt.dayofweek
test_df['dow'] = test_df['visit_date'].dt.dayofweek
aggregation = {'visitors' :{'total_visitors' : 'median'}}

#%% Group by id and day of week - Median of the visitors is taken
agg_data = train.groupby(['air_store_id', 'dow']).agg(aggregation).reset_index()
agg_data.columns = ['air_store_id','dow','visitors']
agg_data['visitors'] = agg_data['visitors']

#%% Create the first intermediate submission file:
merged = pd.merge(test_df, agg_data, how='left', left_on=[
    'store_id','dow'], right_on=['air_store_id','dow'])
final = merged[['id','visitors']]
final.fillna(0, inplace=True)

#%% originally from this kernel:
date_info = pd.read_csv('date_info.csv')
air_visit_data = pd.read_csv('air_visit_data.csv')
sample_submission = pd.read_csv('sample_submission.csv')
#%% remove weekends from holidays
weekend_hdays = date_info.apply(
    (lambda x:(x.day_of_week=='Sunday' or x.day_of_week=='Saturday') 
    and x.holiday_flg==1), axis=1)
date_info.loc[weekend_hdays, 'holiday_flg'] = 0
date_info['weight'] = (date_info.index + 1) / len(date_info) # why ????????? Lower weight to the days that are far away/less relevant
#%%
visit_data = air_visit_data.merge(
    date_info, left_on='visit_date', right_on='calendar_date', how='left')
visit_data.drop('calendar_date', axis=1, inplace=True)
visit_data['visitors'] = visit_data.visitors.map(pd.np.log1p)
#%%
wmean = lambda x:( (x.weight * x.visitors).sum() / x.weight.sum() )
visitors = visit_data.groupby(['air_store_id', 'day_of_week', 'holiday_flg']).apply(wmean).reset_index()
visitors.rename(columns={0:'visitors'}, inplace=True) 
#%%
sample_submission['air_store_id'] = sample_submission.id.map(
    lambda x: '_'.join(x.split('_')[:-1]))
sample_submission['calendar_date'] = sample_submission.id.map(lambda x: x.split('_')[2])
sample_submission.drop('visitors', axis=1, inplace=True)
sample_submission = sample_submission.merge(date_info, on='calendar_date', how='left')
sample_submission = sample_submission.merge(
    visitors, on=['air_store_id', 'day_of_week', 'holiday_flg'], how='left')

#%% fill missings with (air_store_id, day_of_week)
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[visitors.holiday_flg==0], on=(
        'air_store_id', 'day_of_week'), how='left')['visitors_y'].values

#%%% fill missings with (air_store_id)
missings = sample_submission.visitors.isnull()
sample_submission.loc[missings, 'visitors'] = sample_submission[missings].merge(
    visitors[['air_store_id', 'visitors']].groupby('air_store_id').mean().reset_index(), 
    on='air_store_id', how='left')['visitors_y'].values
    
sample_submission['visitors'] = sample_submission.visitors.map(pd.np.expm1)
sample_submission = sample_submission[['id', 'visitors']]
final['visitors'][final['visitors'] ==0] = sample_submission['visitors'][final['visitors'] ==0]
sub_file = final.copy()

#%% Arithmetric Mean 
sub_file['visitors'] = np.mean([final['visitors'], sample_submission['visitors']], axis = 0)
sub_file.to_csv('sub_math_mean.csv', index=False)

## Geometric Mean  
sub_file['visitors'] = (final['visitors'] * sample_submission['visitors']) ** (1/2)
sub_file.to_csv('sub_geo_mean.csv', index=False)

## Harmonic Mean 
sub_file['visitors'] = 2/(1/final['visitors'] + 1/sample_submission['visitors'])
sub_file.to_csv('sub_hrm_mean.csv', index=False)