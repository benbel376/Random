'''
This python module orchestrates data fetching, scoring rancking and generating bidding models(bid list)
in a sequential flow. 
Strart from main() method call and walk your way through the code.
'''

import argparse
from email.policy import default
import os, sys
import gc as garbageCollector
import pandas as pd
import shutil

from time import sleep
from datetime import datetime, timedelta, date

import pyludio.adutils as adu


from helpers.mutual_info import MutulInformation as MIFO
from bidding_algo import BiddingAlgo
from utils.algosbot import AlgoBot
from utils.rds import get_row_by_key
from utils.mlflow import scores_are_different
from ttd.algos_common import *
from fpflow import fpflow


class AutoModelling():

    def __init__(self, days: int = 0, last_date: str = None, specific_campaign:dict = None, testing = False):

        self.days: int = days
        self.today: str = date.today().isoformat()  # YYYY-MM-DD
        self.end_date: str = last_date or self.today

        self.start_date = (adu.get_date(self.end_date) -
                           timedelta(self.days)).date().isoformat()
        self.tm = None

        self.specific_campaign = specific_campaign
        self.pipelineTimeRecord = {}
        self.testing = testing

    def send_slack_notification(self, message):
        notifier = AlgoBot(channel="#only-app-test",
                           purpose="NormalMessage")  # bidding-algo

        notifier.post_block(message_content=message,
                            reference_text="Model update for all running campaigns")

    def info(self, message):
        print()
        print(message)
        print()

    def get_running_campaigns(self):
        status_running = get_row_by_key(
            'campaign', 'algo_status', 'Running', return_multiple_rows=True)
        status_empty = get_row_by_key(
            'campaign', 'algo_status', '', return_multiple_rows=True)
        status_paused = get_row_by_key(
            'campaign', 'algo_status', 'Paused', return_multiple_rows=True)
        status_no_bidlist = get_row_by_key(
            'campaign', 'algo_status', 'No Bidlist Associated', return_multiple_rows=True)
        # No Bidlist Associated

        running_campaigns = [*status_running, *status_empty, *status_paused, *status_no_bidlist]
        # running_campaigns = [*status_running, *status_paused]

        return running_campaigns

    def get_advertiser_campaigns(self) -> dict:
        running_campaigns = self.get_running_campaigns()
        advertisers = {}

        for campaign in running_campaigns:
            advertisers.setdefault(campaign['advertiser_id'], []).append(
                campaign['campaign_id'])

        return advertisers

    def get_minimum_start_date(self, advertiserId:str) -> str:
        running_campaings = self.get_running_campaigns()

        specific_campigns = filter(lambda x: x.get('advertiser_id') == advertiserId, running_campaings)
        start_dates = list(map(lambda x: x.get('start_date'), specific_campigns))
        
        minimum_start_date = min(start_dates)
        minimum_start_date = datetime.strptime(minimum_start_date, '%Y-%m-%d')
        
        return min(start_dates)

    def get_scored_data(self, df_scaled, kpi_args, kpi_list):
        scored_df = self.tm.score_data(
            df_scaled, kpis=kpi_args, KPI=kpi_list, beta=False, clean=True, boxprice=True)
        return scored_df

    def save_model(self, adjusted_bid):
        saved_model = adjusted_bid.save_model(
            model_path=self.tm.model_path, s3folder=self.tm.s3folder)
        return saved_model

    def upload_model(self):
        res =  self.tm.upload_bidlistfile(bidlistfile=self.tm.model_path,dryrun=False)
        return res

    def timescaled_kpi(self, date, start_date, end_date, kpi, var):

        if date is not None and isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()

        if start_date is not None and isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()

        if end_date is not None and isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()

        x = (date - start_date).days
        k = (end_date - start_date).days

        if var == 1:  # linear growth
            scaled_kpi = (x + 1) / (k + 1)

        if var == 2:  # exponential growth
            scaled_kpi = (1 + (1 / max([1, k])))**x

        return (scaled_kpi * kpi)

    def get_scaled_kpis(self, df, var=2):

        df.LogEntryTime = pd.to_datetime(df.LogEntryTime)
        df.engagement.astype(str).astype(float)

        start_date = df.LogEntryTime.min().date()
        end_date = df.LogEntryTime.max().date()

        if 'engagement' in df.columns:
            df.loc[:, ['engagement']] = df[['LogEntryTime', 'engagement']]\
                .apply(lambda eng: self.timescaled_kpi(eng.LogEntryTime.date(),
                                                       start_date,
                                                       end_date,
                                                       eng.engagement,
                                                       var), axis=1
                       )

        if 'click' in df.columns:
            df.loc[:, ['click']] = df[['LogEntryTime', 'click']]\
                .apply(lambda click: self.timescaled_kpi(click.LogEntryTime.date(),
                                                         start_date,
                                                         end_date,
                                                         click['click'],
                                                         var), axis=1
                       )

        if 'viewable' in df.columns:
            df.loc[:, ['viewable']] = df[['LogEntryTime', 'viewable']]\
                .apply(lambda view: self.timescaled_kpi(view.LogEntryTime.date(),
                                                        start_date,
                                                        end_date,
                                                        view.viewable,
                                                        var), axis=1
                       )

        if 'video-end' in df.columns:
            df.loc[:, ['video-end']] = df[['LogEntryTime', 'video-end']]\
                .apply(lambda video: self.timescaled_kpi(video.LogEntryTime.date(),
                                                         start_date,
                                                         end_date,
                                                         video['video-end'],
                                                         var), axis=1
                       )

        return df

    def get_kpis(self):

        kpi_list = [{'ER%': 'ER', 'ER %': 'ER'}.get(
            x, x) for x in self.tm.campaign_params['kpi'].replace(' ', '').split(',')]
        if 'Viewability' in kpi_list:
            kpi_list.remove('Viewability')
            kpi_list.append('VR')

        kpi_args = {}

        if 'ER' in kpi_list:
            kpi_args['engagement'] = 'sum'

        if 'CTR' in kpi_list:
            kpi_args['click'] = 'sum'

        if 'VTR' in kpi_list:
            kpi_args['video-end'] = 'sum'
            kpi_args['video-start'] = 'sum'

        if 'VR' in kpi_list:
            kpi_args['viewable'] = 'sum'

        if 'Conversions' in kpi_list:
            kpi_args['converted'] = 'sum'
            kpi_list.remove('Conversions')
            kpi_list.append('CPA')

        return kpi_list, kpi_args
    
    def drop_score_KPI(self,df):
        '''to return to old type of modelling drop score_ columns'''
        cols = df.columns[df.columns.str.contains('score_')]
        return df.drop(columns=cols)


    def adjust_bid(self, dfw, kpi_list):

        def drop_score_KPI(df):
            '''to return to old type of modelling drop score_ columns'''
            cols = df.columns[df.columns.str.contains('score_')]
            return df.drop(columns=cols)

        # create model and save it to json file
        lrate = {'ER': 0, 'eCTR': 0, 'CTR': 0, 'VTR': 0,
                 'VR': 0, 'CPA': 0}  # KPI weight and learning rate
        kwfit = {'costcol': 'price_median'}

        # give strong weight to KPIs reqired
        for kpi in kpi_list:
            lrate[kpi] = 1
        print(lrate)
        multikpi = False

        if multikpi:
            dfx = dfw
            # Start position for the model parameter optimisation. It should be #kpi + 1
            xstart = [0.1 for x in kpi_list]
        else:
            dfx = drop_score_KPI(dfw)
            xstart = None

        adml = self.tm.make_model_basin(dfx, bfid=3, verbose=2, niter=100, T=2,
                                        xstart=xstart, wkpi=lrate, step_size=2,
                                        kwfit=kwfit)

        return adml

    def clean_modeling_dataframe(self, df):
        os_api_list = pd.read_csv("data/OS_mapping.csv")
        os_api_columns_list = os_api_list["API"].to_list()

        final_df = df.copy()
        print("--------------- Bad sites --------------")
        test_final_df = final_df[final_df['Site'].str.len() < 3]
        if not(test_final_df.empty):
            print(test_final_df["Site"])

        final_df = final_df[final_df['Site'].str.len() >= 3]
        print("----------------------------------------")
        
        print("---------- Out of range OS ------------")
        final_df["OS"] = final_df["OS"].apply(int) # some values come in quotes and need to be converted.
        test_final_df = final_df[~final_df["OS"].isin(os_api_columns_list)]
        
        if not(test_final_df.empty):
            print(test_final_df["OS"])
        
        final_df = final_df[final_df["OS"].isin(os_api_columns_list)]
        print("----------------------------------------")
        
        return final_df

    def get_memory_used(self):
        try:
            _, used_memory, _ = map(int, os.popen("free -t -m").readlines()[-1].split()[1:])
        except IndexError:
            print("Command free Might not be found on the setup")
            used_memory = 0
        return used_memory

    def shall_use_emr(self):
        '''
         sys.exit(0) -> means we only have one advertiser
         sys.exit(1) -> means we have more than one and we could use emr
         This function is used in airflow dag to decide weather we should use
         EMR or local setup
        '''
        advertisers = AutoModelling().get_advertiser_campaigns()
        if(len(advertisers.keys()) > 1):
            # We need to use AWS-EMR
            os.environ['USE_EMR'] = "yes"
            sys.exit(0)
        else:
            # We can run on local machine
            os.environ['USE_EMR'] = "no"
            sys.exit(111)

    def main(self):
        self.info("|************** [ MODEL UPLOAD STARTED ] **************|")
        modelStartTime = datetime.now()
        self.pipelineTimeRecord['algo_Memory_Usage'] = [self.get_memory_used()]  # unknown

        advertiser_campaign_mapping: dict = self.specific_campaign or self.get_advertiser_campaigns()
        print('---- Total Campaign to Model ----')
        print(advertiser_campaign_mapping)
        print("---------------------------------")
        TOTAL_RUNNING_ADVERTISERS: int = len(
            advertiser_campaign_mapping.keys())

        brand_names: list = ['\n']

        for index, advertiser in enumerate(advertiser_campaign_mapping.keys()):
            campaign_ids: list = advertiser_campaign_mapping[advertiser]
            self.start_date = self.start_date if self.days else self.get_minimum_start_date(advertiser)

            self.info(
                f"AdvertiserId => {advertiser} : Progress => {(index + 1)} / {(TOTAL_RUNNING_ADVERTISERS)} ")
            self.info(
                f'{"Camapaign" if len(campaign_ids) == 1 else "Campaigns"} => {campaign_ids}')

            temp_tm = BiddingAlgo(cid=campaign_ids[0],start_date=self.start_date,end_date=self.end_date)
             
            # kwie['subdir'] = 'java_test' # helpfull to test dates, rather than saving to actual work space
            # kwie["logtype"] = ['imp']

            self.info(f"FETCHING DATA ...")
            
            fetchingStartTime = datetime.now()
            #Fetch data using pipeline and entity
            algoscommon=AlgosCommonTddModel(algos=temp_tm)
            # Fetch data once per advertiser
            
            file = f"{advertiser}-{self.start_date}-{self.end_date}.csv"
            if (self.testing and os.path.exists(file)):
                dfall = pd.read_csv(file)
            else:
                dfall, _, _ = algoscommon.fetch_data()
                dfall.to_csv(file, index=False)
            if dfall.empty:
                    # No data is fetched, or campaign was not running on those dates
                    # and don't have data to process
                    self.info(
                        f"Campaign => {campaign_ids[0]} was not running on the date ranges {self.start_date} - {self.end_date}")
                    continue
            
            fetchingEndTime = datetime.now()
            self.pipelineTimeRecord['auto_model_Fetching'] = str(fetchingEndTime - fetchingStartTime)
            self.pipelineTimeRecord['algo_Memory_Usage'].append(self.get_memory_used())
            self.pipelineTimeRecord['auto_model_DataFrameSize'] = str(dfall.shape)

            dfall = self.clean_modeling_dataframe(dfall)
            
            for campaign_id in campaign_ids:

                self.tm = BiddingAlgo(cid=campaign_id, start_date=self.start_date,
                                    end_date=self.end_date)
                self.tm.pipelineTimeRecord = self.pipelineTimeRecord.copy()
               
                #Fetch data using pipeline and entity
                algoscommon=AlgosCommonTddModel(algos=self.tm)
                
                campaign_values = [self.tm.entity.campaign_id, self.tm.entity.algo['algo_campaign_id']]

                cpcv_campaigns = ['yvvviij', 'mvx3w0z']
                df = dfall
                if campaign_id not in cpcv_campaigns:
                    df = dfall.loc[dfall['CampaignId'].isin(campaign_values)]
                
                self.info("GENERATE TRAINING FEATURES WITH MUTUAL INFORAMTION ....")
                mutualInfo = MIFO(df, algoscommon.get_algos_common_attributes()["kpi"])
                training_columns = mutualInfo.top4_columns()
                
                self.tm.pipelineTimeRecord['auto_model_MutualInfo'] = mutualInfo.timeTook
                self.tm.pipelineTimeRecord['algo_Memory_Usage'].append(self.get_memory_used())


                self.info("SCALLING KPIS ...")
                scallingStartTime = datetime.now()
                #
                df_scaled = self.get_scaled_kpis(df)
                scallingEndTime = datetime.now()
                #
                self.tm.pipelineTimeRecord['auto_model_Scalling'] = str(scallingEndTime - scallingStartTime)
                self.tm.pipelineTimeRecord['algo_Memory_Usage'].append(self.get_memory_used())

                self.info("SCORING DATA ..")
                self.tm.pipeline_config(**{"ColNames": training_columns}) # set dynamic columns
                scored_df = self.tm.score_data(df_scaled, KPI = self.tm.entity.kpi, beta=False, clean=True, boxprice=True)
                self.tm.pipelineTimeRecord['algo_Memory_Usage'].append(self.get_memory_used())


                dfx = self.drop_score_KPI(scored_df)
                adml = self.tm.build_model(dfx, bfid=3,verbose=2, basin_iterations=100, temperature=2, xstart=None,step_size=2)
                self.info(
                    f"ADJUSTED BID ...SAVING MODEL to => {self.tm.model_path} : S3 => {self.tm.s3folder}  ...")
                self.tm.pipelineTimeRecord['algo_Memory_Usage'].append(self.get_memory_used())

                self.info("UPLOAD MODEL ...")
                self.tm.upload_bidlistfile(bidlistfile=self.tm.model_path,dryrun=False)
                self.tm.pipelineTimeRecord['algo_Memory_Usage'].append(self.get_memory_used())

                self.info("SAVING MODEL DATA TO MLFLOW ..")
                # fpflow(self.tm, df, scored_df)

                brand_names.append(self.tm.entity.parameters['campaign_params']['brandname'])

                self.info(
                    f"************** [ COMPLETED CAMPAIGN => {campaign_id}] **************")
                
                # clean the variables for garbage
                del self.tm
                del df
                del df_scaled
                del scored_df
                garbageCollector.collect()

            # clean the variables for garbage
            del dfall
            del temp_tm
            garbageCollector.collect()

        totalModelingTime = str(datetime.now() - modelStartTime).split('.')[0]

        if len(brand_names) > 1:
            self.info("NOTIFING ON SLACK ..")
            brand_names = '\n'.join(brand_names)
            self.send_slack_notification(brand_names)
            self.send_slack_notification(
                f"\n|****  [ MODEL UPLOAD COMPLETED ] For {self.today} , Total Duration { totalModelingTime } ****|")
            print("Message_content => ", brand_names)
        else:
            self.send_slack_notification(
                f"|**** [ No Algo Campaigns To Model ] For {self.today} ****")

        self.info(f"\n|**** [ MODEL UPLOAD COMPLETED ] For {self.today} , Total Duration { totalModelingTime } ****|")

if __name__ == '__main__':
    # Construct the argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--days", default=45, help="Number of Days to fetch")
    parser.add_argument("-ld","--lastDate", default=None, help="user defined end_date for analysis")
    parser.add_argument("-adId", "--advertiserId", default=None, help="Advertiser Id")
    parser.add_argument("-cId", "--campaignId", default=None, help="Campaign Id")
    parser.add_argument("-emr", "--testForEmr", default=False, help="run test to run cluster or not")
    parser.add_argument("-t", "--testing", default=False, help="run automodeling on testing mode ")
    args = vars(parser.parse_args())
    
    days = int(args['days'])
    last_date = args['lastDate']
    advertiserId = args['advertiserId']
    campaignId = args['campaignId']
    specific_campaign = {advertiserId:[campaignId]} if bool(advertiserId) else None
    # specific_campaign = {"tq89q6u":['yvvviij', 'mvx3w0z']}
    testing = bool(args["testing"])

    auto = AutoModelling(days=days, last_date=last_date, specific_campaign=specific_campaign, testing=testing)
    if(bool(args['testForEmr'])):
        print('-- test wheather to run on emr or local machine ...')
        auto.shall_use_emr()
    else:
        print("--running bidding algo")
        auto.main()
