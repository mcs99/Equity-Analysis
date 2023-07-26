'''
Notes:
- A SimFin subscription and API is required to use this tool
- To get a SimFin subscription, please purchase from: https://www.simfin.com/en/
- Once you have a subscription, please enter your key in the following line of code:
    - sf.set_api_key('API KEY HERE')

Purpose:
- Simply enter a ticker to:
- Output relevant statistical tables and supporting visuals
- Perform modeling and quick calculations
'''
import simfin as sf
import pandas as pd
import yfinance as yf
import datetime
from random import randint
import numpy as np
from fredapi import Fred

import openai

sf.set_api_key('XXX') # Set your SimFin+ API-key for downloading data 
sf.set_data_dir('~/simfin_data/') # Set the local directory where data-files are stored, change this to be this folder if possible



class Tables:
    def __init__(self, ticker):
        self.ticker = ticker

        self.all_companies = sf.load_companies().reset_index()
        self.company = self.all_companies[self.all_companies['Ticker'] == self.ticker]
        self.industry_id = self.company['IndustryId'].iloc[0]
        industries = sf.load_industries().reset_index()
        self.co_industry = industries[industries['IndustryId'] == self.industry_id]
        self.str_industry = self.co_industry['Industry'].iloc[0]

        self.derived = sf.load_derived(variant='annual', market='us').reset_index()
        self.balance = sf.load_balance(variant='annual', market='us').reset_index()
        self.income = sf.load_income(variant='annual', market='us').reset_index()
        self.cashflow = sf.load_cashflow(variant='annual', market='us').reset_index()
        self.financials_dict = {'balance':self.balance, 'derived':self.derived, 'income':self.income, 'cashflow':self.cashflow}

        historicals_start = 2018
        historicals_end = 2022
        self.historical_dates = [i for i in range(historicals_start, historicals_end+1)]

        projection_start = historicals_end + 1
        projection_end = projection_start + 7
        self.projection_dates = [i for i in range(projection_start, projection_end+1)]

        # check folder and make folder here
        return None

    def get_financials(self):
        industry_companies = self.all_companies[self.all_companies['IndustryId']==self.industry_id]
        industry_companies_lst = list(industry_companies['Ticker'].unique())

        final_ratios = []

        for i in self.financials_dict.keys():
            df = self.financials_dict[i]
            doc = i

            # df = df.merge(price, on='Ticker')

            df = df[(df['Ticker'].isin(industry_companies_lst)) & (df['Fiscal Year'].isin(self.historical_dates))] # .reset_index().T
            self.financials_dict[doc] = df
            # df.to_csv('pulled_data/{}-{}.csv'.format(doc, industry))

        for t in industry_companies_lst:
            for y in self.historical_dates:
                ratios = self.get_ratios(t, y, self.financials_dict)
                final_ratios.append(ratios)

        ratio_df = pd.DataFrame(final_ratios)
        ratio_df = ratio_df.sort_values(by=['ticker', 'year'])
        self.financials_dict['ratios'] = ratio_df

        # print(self.financials_dict)

        return self.financials_dict 

    def get_ratios(self, ticker_sym, year, docs):
        balance = docs['balance']
        balance_cur = balance[(balance['Ticker'] == ticker_sym) & (balance['Fiscal Year'] == year)]
        balance_ltm = balance[(balance['Ticker'] == ticker_sym) & (balance['Fiscal Year'] == year-1)]

        income = docs['income']
        income_cur = income[(income['Ticker'] == ticker_sym) & (income['Fiscal Year'] == year)]
        income_ltm = income[(income['Ticker'] == ticker_sym) & (income['Fiscal Year'] == year-1)]

        cashflow = docs['cashflow']
        cashflow = cashflow[(cashflow['Ticker'] == ticker_sym) & (cashflow['Fiscal Year'] == year)]
        cashflow_ltm = cashflow[(cashflow['Ticker'] == ticker_sym) & (cashflow['Fiscal Year'] == year-1)]
        
        # last_price = price[(price['Ticker'] == self.ticker)]
        
        ratios = {}
        
        ratios['ticker'] = ticker_sym
        ratios['year'] = year
        
        try:
            ratios['current-ratio'] = balance_cur['Total Current Assets'].iloc[0] / balance_cur['Total Current Liabilities'].iloc[0]
        except:
            ratios['current-ratio'] = None
            pass

        try:
            ratios['quick-ratio'] = (balance_cur['Total Current Assets'].iloc[0] - balance_cur['Inventories'].iloc[0]) / balance_cur['Total Current Liabilities'].iloc[0]
        except:
            ratios['quick-ratio'] = None
            pass

        try:
            ratios['cash-ratio'] = balance_cur['Cash, Cash Equivalents & Short Term Investments'].iloc[0] / balance_cur['Total Current Liabilities'].iloc[0]
        except:
            ratios['cash-ratio'] = None
            pass

        try:
            ratios['debt-ratio'] = balance_cur['Total Liabilities'].iloc[0] / balance_cur['Total Assets'].iloc[0]
        except:
            ratios['debt-ratio'] = None
            pass

        try:
            ratios['debt-equity'] = balance_cur['Total Liabilities'].iloc[0] / balance_cur['Total Equity'].iloc[0]
        except:
            ratios['debt-equity'] = None
            pass 

        try:
            ratios['interest-coverage'] = income_cur['Operating Income (Loss)'].iloc[0] / (income_cur['Interest Expense, Net'].iloc[0]*-1)
        except:
            ratios['interest-coverage'] = None
            pass 

        try:
            ratios['asset-turnover'] = income_cur['Revenue'].iloc[0] / ((balance_cur['Total Assets'].iloc[0] + balance_ltm['Total Assets'].iloc[0])/2)
        except:
            ratios['asset-turnover'] = None
            pass

        try:
            ratios['ar-turnover'] = income_cur['Revenue'].iloc[0] / ((balance_cur['Accounts & Notes Receivable'].iloc[0] + balance_ltm['Accounts & Notes Receivable'].iloc[0])/2)
        except:
            ratios['ar-turnover'] = None
            pass

        try:
            ratios['inventory-turnover'] = income_cur['Revenue'].iloc[0] / ((balance_cur['Inventories'].iloc[0] + balance_ltm['Inventories'].iloc[0])/2)
        except:
            ratios['inventory-turnover'] = None
            pass

        try:
            ratios['gross-margin'] = income_cur['Gross Profit'].iloc[0] / income_cur['Revenue'].iloc[0]
        except:
            ratios['gross-margin'] = None
            pass

        try:
            ratios['operating-margin'] = income_cur['Operating Income (Loss)'].iloc[0] / income_cur['Revenue'].iloc[0]
        except:
            ratios['operating-margin'] = None
            pass

        try:
            ratios['roa'] = income_cur['Net Income'].iloc[0] / balance_cur['Total Assets'].iloc[0]
        except:
            ratios['roa'] = None
            pass

        try:
            ratios['roe'] = income_cur['Net Income'].iloc[0] / balance_cur['Total Equity'].iloc[0]
        except:
            ratios['roe'] = None
            pass

        try:
            ratios['div-yield'] = balance_cur['Dividend'].iloc[0]
        except:
            ratios['div-yield'] = None
            pass


        try:
            ratios['rev-growth'] = (income_cur['Revenue'].iloc[0] - income_ltm['Revenue'].iloc[0]) / income_ltm['Revenue'].iloc[0]
        except:
            ratios['rev-growth'] = None
            pass

        try:
            ratios['op-income-growth'] = (income_cur['Operating Income (Loss)'].iloc[0] - income_ltm['Operating Income (Loss)'].iloc[0]) / income_ltm['Operating Income (Loss)'].iloc[0]
        except:
            ratios['op-income-growth'] = None
            pass

        try:
            ratios['income-growth'] = (income_cur['Net Income'].iloc[0] - income_ltm['Net Income'].iloc[0]) / income_ltm['Net Income'].iloc[0]
        except:
            ratios['income-growth'] = None
            pass


        return ratios

    def horizon_score(self, ratios):
        print(ratios.columns)
        results_lst = []
        
        for year in self.historical_dates:
            print(year)

            new_ratios = ratios[ratios['year'] == year]
            print(new_ratios)

            cur_ratio_mean = new_ratios['current-ratio'].mean()
            cash_ratio_mean = new_ratios['cash-ratio'].mean()
            debt_ratio_mean = new_ratios['debt-ratio'].mean()
            debt_equity_mean = new_ratios['debt-equity'].mean()
            roa_mean = new_ratios['roa'].mean()
            roe_mean = new_ratios['roe'].mean()
            rev_growth_mean = new_ratios['rev-growth'].mean()
            income_growth_mean = new_ratios['income-growth'].mean()
            
            for i in new_ratios.to_dict('records'): # check this code
                results = {}
          
                results['year'] = year

                score = 0
                total = 8
                # print(i)
                results['ticker'] = i['ticker']

                
                if i['current-ratio'] < cur_ratio_mean:
                    score += 0
                elif i['current-ratio'] > cur_ratio_mean:
                    score += 1
                
                if i['cash-ratio'] < cash_ratio_mean:
                    score += 0
                elif i['cash-ratio'] > cash_ratio_mean:
                    score += 1      
                
                if i['debt-ratio'] > debt_ratio_mean:
                    score += 0
                elif i['debt-ratio'] < debt_ratio_mean:
                    score += 1      

                if i['debt-equity'] > debt_equity_mean:
                    score += 0
                elif i['debt-equity'] < debt_equity_mean:
                    score += 1      
                
                if i['roa'] < roa_mean:
                    score += 0
                elif i['roa'] > roa_mean:
                    score += 1      

                if i['roe'] < roe_mean:
                    score += 0
                elif i['roe'] > roe_mean:
                    score += 1      

                if i['rev-growth'] < rev_growth_mean:
                    score += 0
                elif i['rev-growth'] > rev_growth_mean:
                    score += 1      

                if i['income-growth'] < income_growth_mean:
                    score += 0
                elif i['income-growth'] > income_growth_mean:
                    score += 1     

                results['score'] = int(score/total*100)

                results_lst.append(results)

        scores = pd.DataFrame(results_lst)
        
        return scores         
    
    def get_returns(self):

        results = []
        last_yr = max(self.historical_dates)

        for ticker in list(self.all_companies[self.all_companies['IndustryId']==self.industry_id]['Ticker']) + ['^GSPC']:
            print(ticker)
        
            COMPANY = yf.Ticker(ticker)
            price_df = COMPANY.history(period='max')
            price_df['Ticker'] = ticker


            for year in self.historical_dates:
                res = {}
                res['ticker'] = ticker
                res['year'] = year
                try:
                    res['current_price'] = price_df['Close'].iloc[-1]
                    
                    if last_yr != year:
                        res['last_price'] = price_df['Close'].iloc[-252*(last_yr-year)]
                        res['returns'] = (res['current_price'] - res['last_price']) / res['last_price']
                    else:
                        res['last_price'] = price_df['Close'].iloc[-1]
                        res['returns'] = (res['current_price'] - res['last_price']) / res['last_price']

                    results.append(res)
                except:
                    pass

        results_df =pd.DataFrame(results)

        return results_df

    def get_news(self): # Must pay for OpenAI subscription
    #     openai.api_key = 'XXX'
    #     model = 'davinci'

    #     results = []
    #     for ticker in self.all_companies[self.all_companies['IndustryId']==self.industry_id]['Ticker']:
        
    #         COMPANY = yf.Ticker(ticker)
    #         res = {}
    #         news_df = COMPANY.news
            
    #         for x in news_df:
    #             print(x['link'], x['providerPublishTime'], x['type'])

    #             question = "Cann you go to this URL and summarize this article: {}".format(x['link'])

    #             response = openai.Completion.create(
    #                 engine=model,
    #                 prompt=(f"Question: {question}\n"
    #                         "Answer:"
    #                         ),
    #                 max_tokens=100,
    #                 n=1,
    #                 stop=None,
    #                 temperature=0.5
    #             )

    #             answer = response.choices[0].text.split('\n')[0]
    #             print(answer)

            # news_df['Ticker'] = ticker
        
        return
    
    def price_ratios(self):
        results = []
        
        last_yr = max(self.historical_dates)

        for year in self.historical_dates:
            for ticker in list(self.all_companies[self.all_companies['IndustryId']==self.industry_id]['Ticker'].unique()):
            
                COMPANY = yf.Ticker(ticker)
                res = {}
                price_df = COMPANY.history(period='max')
                price_df['Ticker'] = ticker

                try:
                    # for loop here for different years
                    
                    res['ticker'] = ticker
                    res['year'] = year
                    if last_yr != year:
                        res['last_price'] = price_df['Close'].iloc[-252*(last_yr-year)]
                    else:
                        res['last_price'] = price_df['Close'].iloc[-1]
                    res['shares_out'] = self.income[(self.income['Ticker'] == ticker)&(self.income['Fiscal Year'] == year)]['Shares (Basic)'].iloc[-1]
                    res['earnings'] = self.income[(self.income['Ticker'] == ticker)&(self.income['Fiscal Year'] == year)]['Net Income'].iloc[-1]
                    res['sales'] = self.income[(self.income['Ticker'] == ticker)&(self.income['Fiscal Year'] == year)]['Revenue'].iloc[-1]
                    res['book_val'] = self.balance[(self.balance['Ticker'] == ticker)&(self.balance['Fiscal Year'] == year)]['Total Equity'].iloc[-1]
                    res['pe'] = res['last_price'] / (res['earnings']/res['shares_out'])
                    res['ps'] = res['last_price'] / (res['sales']/res['shares_out'])
                    res['pb'] = res['last_price'] / (res['book_val']/res['shares_out'])
                    print(res)
                    results.append(res)
                except:
                    pass
            
        return pd.DataFrame(results)
    
    def get_technicals(self):
        'dataset of all prices + S&P prices, with momemntum, volatility, and other ratios calculated as additional cols.'
        # first, make s&p/stocks price dataset 
        
        df_lst = []
        
        for t in  [self.ticker, '^GSPC']:
            print(t)
        
            COMPANY = yf.Ticker(t)
            price_df = COMPANY.history(period='max').reset_index()
            print(price_df)
            price_df['Date'] = price_df['Date'].dt.tz_localize(None)
            print(price_df.columns)
            price_df['Ticker'] = t
            # price_df['Date'] = price_df['Date'].tz_localize(None)

            price_df['T EWM'] = price_df['Close'].ewm(span=26, adjust=False).mean()
            price_df['T+x EWM'] = price_df['Close'].ewm(span=52, adjust=False).mean()
            price_df['Signal Line'] = price_df['T+x EWM'] - price_df['T EWM']
            price_df['MACD'] = price_df['T+x EWM'] - price_df['T EWM']
            price_df['Signal'] = price_df['MACD'].ewm(span=9, adjust=False).mean()

            
            price_df['Change'] = price_df['Close'].diff()
            price_df['Change_Up'] = price_df['Change']
            price_df['Change_Down'] = price_df['Change']
            price_df.loc[price_df['Change']>0, 'Change_Down'] = 0
            price_df.loc[price_df['Change']<0, 'Change_Up'] = 0
            avg_up = price_df['Change_Up'].rolling(14).mean()
            avg_down = price_df['Change_Down'].rolling(14).mean().abs()
            price_df['Price RSI'] = 100 * avg_up / (avg_up + avg_down)


            price_df['100_St_Dev'] = price_df['Close'].rolling(100).std()
            price_df['100_Day_MA'] = price_df['Close'].rolling(100).mean()
            price_df['Bollinger_Top'] = price_df['Close'].rolling(100).std() * -2 +  price_df['100_Day_MA']
            price_df['Bollinger_Bottom'] = price_df['Close'].rolling(100).std() * 2 + price_df['100_Day_MA']

            price_df["OBV"] = (np.sign(price_df["Close"].diff()) * price_df["Volume"]).fillna(0).cumsum() # check of this should be made rolling somehow

            # price_df['Down Change'] = 

            df_lst.append(price_df)

        df = pd.concat(df_lst)
        print(df)



        # To Do
        # s&p correlation coeff.: https://www.stockopedia.com/learn/charts-technical-analysis/correlation-coefficient-463013/ (save for end once dfs are merged)

        # Done
        # macd: https://www.stockopedia.com/learn/charts-technical-analysis/macd-462943/
        # rsi: https://www.stockopedia.com/learn/charts-technical-analysis/rsi-462948/
        # standard dev: https://www.stockopedia.com/learn/charts-technical-analysis/standard-deviation-463003/     
        # bollinger bands: https://www.stockopedia.com/learn/charts-technical-analysis/bollinger-bands-462688/
        # on-balance volume: https://blog.elearnmarkets.com/volume-indicator/        


        return df

    def score(self):
        '''here, get all data considered, and final scores'''
        return
    
    def economics(self): 

        fred = Fred(api_key='XXX')
        desired_series = {
            'Federal Funds Effective Rate': 'DFF', 
            'T2Y10Y Yield Curve': 'T10Y2Y',
            'CPI': 'CPIAUCSL',
            'PPI': 'PPIACO',
            'Leading Index': 'BBKMLEIX',
            'Coincident Economic Activity': 'USPHCI',
            'Brave Butlers Kelley': 'BBKMCOIX',
            'Sales Growth Expectations': 'ATLSBUSRGEP',
            'Employment Growth Expecations': 'ATLSBUEGEP',
            'Personal Saving Rate': 'PSAVERT',
            'Smoothed Recession Prob': 'RECPROUSM156N',
            'GDP': 'GDP',
            'PCE': 'PCE',
            'Inventories to Sales': 'ISRATIO', # Inventories
            'Avg Hourly Earnings': 'CES0500000003',# Wages
        }
        
        
        row = {}
        row_count = len(desired_series)

        while row_count > 0:
            for i in desired_series:    
                row[i] = fred.get_series(desired_series[i])
                row_count -= 1
        
        df = pd.DataFrame(row).reset_index().fillna(method='ffill')
        df = df.rename(columns={'index':'Date'})
        print(df)

        return df


class DCF:
    def __init__(self, balance, income, ticker):        
        self.balance_new = balance[balance['Ticker'] == ticker]
        self.income_new = income[income['Ticker'] == ticker]

        self.balance_cols = list(pd.read_csv('codifications/balance_cols.csv').columns)
        self.income_cols = list(pd.read_csv('codifications/income_cols.csv').columns)

        self.balance_fy = list(balance['Fiscal Year'].unique())
        self.income_fy = list(income['Fiscal Year'].unique())

        start_hist_yr = 2018
        end_hist_yr = 2022
        project_len = 5
        self.historical_yrs = [i for i in range(start_hist_yr, end_hist_yr+1)]
        self.project_yrs= [i for i in range(end_hist_yr, end_hist_yr+project_len+1)]
        return
    
    def create_projections(self):
        bsis  = self.balance_new.add_suffix('_balance').join(self.income_new.add_suffix('_income'), how='outer')

        bsis = bsis.fillna(0)
        bsis = self.calc_balances(sheet=bsis, purpose='plug') # this should prob be moved up to not mess up the value derivation

        bsis['RevChange'] = bsis['Revenue_income'].pct_change()
        bsis['InvChange'] = bsis['Inventories_balance'].pct_change() 
        bsis['PpeChange'] = bsis['Property, Plant & Equipment, Net_balance'].pct_change() 
        bsis['ApChange'] = bsis['Payables & Accruals_balance'].pct_change()
        bsis['ArChange'] = bsis['Accounts & Notes Receivable_balance'].pct_change()

        # # Balance figures (plug cash and equity after cash flow and income projection)
        bsis['ar_turn'] = bsis['Revenue_income'] / ((bsis['Accounts & Notes Receivable_balance'] + (bsis['Accounts & Notes Receivable_balance']*(1-bsis['ArChange'])))/2) # net credit sales / avg acct rec
        bsis['inv_turn'] = (bsis['Cost of Revenue_income']*-1) / ((bsis['Inventories_balance'] + (bsis['Inventories_balance']*(1-bsis['InvChange'])))/2) # cogs / avg inv
        bsis['ppe_pct_rev'] =  bsis['Property, Plant & Equipment, Net_balance'] / bsis['Revenue_income'] 
        bsis['lta_ar_pct_rev'] = bsis['Long Term Investments & Receivables_balance'] / bsis["Revenue_income"]
        bsis['other_lta_pct_rev'] = bsis['Other Long Term Assets_balance'] / bsis['Revenue_income']
        bsis['days_payable'] = ((bsis['Payables & Accruals_balance'] + (bsis['Payables & Accruals_balance']*(1-bsis['ApChange'])))/2) / (bsis['Cost of Revenue_income']*-1)*365
        bsis['short_debt_pct_rev'] = bsis['Short Term Debt_balance'] / bsis['Revenue_income']
        bsis['long_debt_pct_rev'] = bsis['Long Term Debt_balance'] / bsis['Revenue_income']
        bsis['share_cap_pct_rev'] = bsis['Share Capital & Additional Paid-In Capital_balance'] / bsis['Revenue_income']
        bsis['treas_stock_pct_rev'] = bsis['Treasury Stock_balance'] / bsis['Revenue_income']

        # Income figures
        bsis['revenue_growth'] = bsis['RevChange'] 
        bsis['margin'] = (bsis['Revenue_income'] + bsis['Cost of Revenue_income']) / bsis['Revenue_income']
        # bsis['opex_pct_rev'] = bsis['Operating Expenses_income'] / bsis['Revenue_income']
        bsis['sga_pct_rev'] = bsis['Selling, General & Administrative_income'] / bsis['Revenue_income']
        bsis['rd_pct_rev'] = bsis['Research & Development_income'] / bsis['Revenue_income']
        bsis['da_pct_rev'] = bsis['Depreciation & Amortization_income'] / bsis['Revenue_income']
        bsis['non_op_inc_pct_rev'] = bsis['Non-Operating Income (Loss)_income'] / bsis['Revenue_income']
        bsis['interest_exp_pct_debt'] = bsis['Interest Expense, Net_income'] / (bsis['Long Term Debt_balance'] + bsis['Short Term Debt_balance'])
        bsis['effective_tax'] = bsis['Income Tax (Expense) Benefit, Net_income'] / bsis['Pretax Income (Loss)_income']
        bsis['ab_gains_pct_rev'] = bsis['Abnormal Gains (Losses)_income'] / bsis['Revenue_income']
        bsis['ext_gains_pct_rev'] = bsis['Net Extraordinary Gains (Losses)_income'] / bsis['Revenue_income']
        return bsis

    def project_inputs(self, sheet):
        '''simply, iterate over the projection years (years), and perform neccesary calcs 
        steps to completion
        ---- quick fixes
        taxes --> save for end
        calculate end balances
        ---- broad topics below (functions, etc.)
        create cash flows
        create ufcf
        calculate dcf inputs
        calculate equity and enterprise (value), eq - net debt
        '''
        sheet = sheet.fillna(0)
        projection_years = self.project_yrs

        for i in projection_years:
            # Revenue
            revenue = sheet[sheet['Fiscal Year_balance']==i-1]['Revenue_income'].iloc[0] * (1+sheet[sheet['Fiscal Year_balance']==i]['RevChange'].iloc[0])
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Revenue_income'] = revenue

            cogs = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * (1-sheet[sheet['Fiscal Year_balance']==i]['margin'].iloc[0]) * -1
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Cost of Revenue_income'] = cogs

            # opex = sheet[sheet['Fiscal Year_balance']==i]['opex_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            # sheet.loc[sheet['Fiscal Year_balance']==i, 'Operating Expenses_income'] = opex

            sga = sheet[sheet['Fiscal Year_balance']==i]['sga_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Selling, General & Administrative_income'] = sga

            rd = sheet[sheet['Fiscal Year_balance']==i]['rd_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Research & Development_income'] = rd

            da = sheet[sheet['Fiscal Year_balance']==i]['da_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Depreciation & Amortization_income'] = da

            # nonop_inc = sheet[sheet['Fiscal Year_balance']==i]['non_op_inc_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            # sheet.loc[sheet['Fiscal Year_balance']==i, 'Non-Operating Income (Loss)_income'] = nonop_inc

            ab_gains = sheet[sheet['Fiscal Year_balance']==i]['ab_gains_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Abnormal Gains (Losses)_income'] = ab_gains        # ab gains
        
            ext_gains = sheet[sheet['Fiscal Year_balance']==i]['ext_gains_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Net Extraordinary Gains (Losses)_income'] = ext_gains       # ext gains

            ar = (sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] / sheet[sheet['Fiscal Year_balance']==i]['ar_turn'].iloc[0] * 2) - sheet[sheet['Fiscal Year_balance']==i-1]['Accounts & Notes Receivable_balance'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Accounts & Notes Receivable_balance'] = ar

            inv = (sheet[sheet['Fiscal Year_balance']==i]['Cost of Revenue_income'].iloc[0]*-1 / sheet[sheet['Fiscal Year_balance']==i]['inv_turn'].iloc[0] *2) - sheet[sheet['Fiscal Year_balance']==i-1]['Inventories_balance'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Inventories_balance'] = inv        

            # calculate capex in cash flows
            ppe = sheet[sheet['Fiscal Year_balance']==i]['ppe_pct_rev'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Property, Plant & Equipment, Net_balance'] = ppe

            lta_ar = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['lta_ar_pct_rev'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Long Term Investments & Receivables_balance'] = lta_ar       
            
            other_lta = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['other_lta_pct_rev'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Other Long Term Assets_balance'] = other_lta
            
            payables = (sheet[sheet['Fiscal Year_balance']==i]['days_payable'].iloc[0] / 365 * (sheet[sheet['Fiscal Year_balance']==i]['Cost of Revenue_income'].iloc[0]*-1) * 2 - sheet[sheet['Fiscal Year_balance']==i-1]['Payables & Accruals_balance'].iloc[0])
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Payables & Accruals_balance'] = payables    

            st_debt = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['short_debt_pct_rev'].iloc[0] 
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Short Term Debt_balance'] = st_debt          
            lt_debt = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['long_debt_pct_rev'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Long Term Debt_balance'] = lt_debt
            
            int_exp = sheet[sheet['Fiscal Year_balance']==i]['interest_exp_pct_debt'].iloc[0] * (sheet[sheet['Fiscal Year_balance']==i]['Short Term Debt_balance'].iloc[0] + sheet[sheet['Fiscal Year_balance']==i]['Long Term Debt_balance'].iloc[0])
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Interest Expense, Net_income'] = int_exp

            share_cap = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['share_cap_pct_rev'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Share Capital & Additional Paid-In Capital_balance'] = share_cap        
            
            treas_stock = sheet[sheet['Fiscal Year_balance']==i]['Revenue_income'].iloc[0] * sheet[sheet['Fiscal Year_balance']==i]['treas_stock_pct_rev'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance']==i, 'Treasury Stock_balance'] = treas_stock        

        # print(sheet.iloc[:, 0:15])

        sheet_inc = self.net_inc_ret_earn(sheet=sheet, years=projection_years)
        sheet_final = self.cashflows(sheet_inc, projection_years)
        
        self.calc_balances(sheet = sheet_final, purpose='check')

        # balance_check = return true false, if true, export, false, raise error
        sheet_final.to_csv('updated-sheet.csv')

        sims = self.dcf_calcs_sim(projection_years, sheet_final, .034, 1.7, .11)

        return pd.DataFrame(sims, columns=['intrinsic_value'])
        # print(sims)
        # plt.hist(x=sims, bins=20, range=[-600, 600])
        # plt.show()

    def net_inc_ret_earn(self, sheet, years):

        for i in years:

            gross_profit = sheet[sheet['Fiscal Year_balance'] == i]['Revenue_income'].iloc[0] + sheet[sheet['Fiscal Year_balance'] == i]['Cost of Revenue_income'].iloc[0]
            opex = sheet[sheet['Fiscal Year_balance'] == i]['Selling, General & Administrative_income'].iloc[0] + sheet[sheet['Fiscal Year_balance'] == i]['Depreciation & Amortization_income'].iloc[0] +  sheet[sheet['Fiscal Year_balance'] == i]['Research & Development_income'].iloc[0] 
            op_income = gross_profit + opex

            pretax_income_adj = op_income + sheet[sheet['Fiscal Year_balance'] == i]['Interest Expense, Net_income'].iloc[0]
            
            pretax_income = pretax_income_adj + sheet[sheet['Fiscal Year_balance'] == i]['Pretax Income (Loss)_income'].iloc[0]
            # sheet.loc[sheet['Fiscal Year_balance']==i, 'Pretax Income (Loss)_income'] = pretax_income

            tax = sheet[sheet['Fiscal Year_balance'] == i]['effective_tax'].iloc[0] * sheet[sheet['Fiscal Year_balance'] == i]['Pretax Income (Loss)_income'].iloc[0]
            # sheet.loc[sheet['Fiscal Year_balance'] == i, 'Income Tax (Expense) Benefit, Net_income'] = tax
            
            income_cont_ops = tax + pretax_income
            
            net_income = income_cont_ops + sheet[sheet['Fiscal Year_balance'] == i]['Abnormal Gains (Losses)_income'].iloc[0]
            sheet.loc[sheet['Fiscal Year_balance'] == i, 'Net Income_income'] = net_income
            sheet.loc[sheet['Fiscal Year_balance'] == i, 'Retained Earnings_balance'] = net_income + sheet[sheet['Fiscal Year_balance'] == i-1]['Retained Earnings_balance'].iloc[0]


    
        return sheet

    def cashflows(self, sheet, years):
        '''accounting cash flow
        # this assumes tax + interest is a cash payment
        cffo: income stmt / current assets / liabs
        = net income + D&A + change in ap - change in ar - change in inv + change short term debt
        cffi: long term assets
        = - Capex - Long term receivables - other lta
        cfff: equity / debt
        = + delt. long term debt balance - delt. change treasury stock
        '''

        for i in years:
            change_ap = sheet[sheet['Fiscal Year_balance'] == i]['Payables & Accruals_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Payables & Accruals_balance'].iloc[0]
            change_ar = sheet[sheet['Fiscal Year_balance'] == i]['Accounts & Notes Receivable_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Accounts & Notes Receivable_balance'].iloc[0]
            change_inv = sheet[sheet['Fiscal Year_balance'] == i]['Inventories_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Inventories_balance'].iloc[0]
            change_stdebt = sheet[sheet['Fiscal Year_balance'] == i]['Short Term Debt_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Short Term Debt_balance'].iloc[0]
            change_ltar = sheet[sheet['Fiscal Year_balance'] == i]['Long Term Investments & Receivables_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Long Term Investments & Receivables_balance'].iloc[0]
            change_olta =  sheet[sheet['Fiscal Year_balance'] == i]['Other Long Term Assets_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Other Long Term Assets_balance'].iloc[0]
            change_ltdebt = sheet[sheet['Fiscal Year_balance'] == i]['Long Term Debt_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Long Term Debt_balance'].iloc[0]
            change_treas = sheet[sheet['Fiscal Year_balance'] == i]['Treasury Stock_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Treasury Stock_balance'].iloc[0]
            change_share_cap = sheet[sheet['Fiscal Year_balance'] == i]['Share Capital & Additional Paid-In Capital_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Share Capital & Additional Paid-In Capital_balance'].iloc[0]
            capex = sheet[sheet['Fiscal Year_balance'] == i]['Property, Plant & Equipment, Net_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i-1]['Property, Plant & Equipment, Net_balance'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i]['Depreciation & Amortization_income'].iloc[0]

            cffo = sheet[sheet['Fiscal Year_balance'] == i]['Net Income_income'].iloc[0] - sheet[sheet['Fiscal Year_balance'] == i]['Depreciation & Amortization_income'].iloc[0] + change_ap - change_ar + change_stdebt - change_inv + change_share_cap
            cffi = - capex - change_ltar - change_olta
            cfff = change_ltdebt - change_treas
            change_cash = cffo +cffi+cfff


            # print(cffo)
            ufcf = cffo - capex
            sheet.loc[sheet['Fiscal Year_balance'] == i, 'Unlevered Free Cash Flow_cashflow'] = ufcf
            sheet.loc[sheet['Fiscal Year_balance'] == i, 'Cash, Cash Equivalents & Short Term Investments_balance'] = sheet[sheet['Fiscal Year_balance'] == i-1]['Cash, Cash Equivalents & Short Term Investments_balance'].iloc[0] + change_cash

        return sheet

    def calc_balances(self, sheet, purpose):
        '''
        calculate end balances
        '''

        if purpose == 'plug':

            curr_assets = sheet['Cash, Cash Equivalents & Short Term Investments_balance'] + sheet['Accounts & Notes Receivable_balance'] + sheet['Inventories_balance']
            currasset_plug = sheet['Total Current Assets_balance'] - curr_assets
            sheet['Accounts & Notes Receivable_balance'] = sheet['Accounts & Notes Receivable_balance'] + currasset_plug

            noncurr_assets = sheet['Property, Plant & Equipment, Net_balance'] + sheet['Long Term Investments & Receivables_balance'] + sheet['Other Long Term Assets_balance']
            nonca_plug  = sheet['Total Noncurrent Assets_balance'] - noncurr_assets
            sheet['Other Long Term Assets_balance'] = sheet['Other Long Term Assets_balance'] + nonca_plug

            # total_assets = curr_assets + noncurr_assets
            
            curr_liabs = sheet['Payables & Accruals_balance'] + sheet['Short Term Debt_balance']
            currliab_plug = sheet['Total Current Liabilities_balance'] - curr_liabs
            sheet['Payables & Accruals_balance'] = sheet['Payables & Accruals_balance'] + currliab_plug


            noncurr_liabs = sheet['Long Term Debt_balance']
            noncl_plug = sheet['Total Noncurrent Liabilities_balance'] - noncurr_liabs
            sheet['Long Term Debt_balance'] = sheet['Long Term Debt_balance'] + noncl_plug

            # total_liabs = curr_liabs + noncurr_liabs

            total_equity = sheet['Share Capital & Additional Paid-In Capital_balance'] - sheet['Treasury Stock_balance'] + sheet['Retained Earnings_balance']
            equity_plug = sheet['Total Equity_balance'] - total_equity
            sheet['Retained Earnings_balance'] = sheet['Retained Earnings_balance'] + equity_plug
            
            return sheet

        elif purpose == 'check': 
            curr_assets = sheet['Cash, Cash Equivalents & Short Term Investments_balance'] + sheet['Accounts & Notes Receivable_balance'] + sheet['Inventories_balance'] # should be same equations as above if stmt
            noncurr_assets = sheet['Property, Plant & Equipment, Net_balance'] + sheet['Long Term Investments & Receivables_balance'] + sheet['Other Long Term Assets_balance']

            total_assets = curr_assets + noncurr_assets
            sheet['Total Assets_balance'] = total_assets

            curr_liabs = sheet['Payables & Accruals_balance'] + sheet['Short Term Debt_balance']
            noncurr_liabs = sheet['Long Term Debt_balance']

            total_liabs = curr_liabs + noncurr_liabs
            sheet['Total Liabilities_balance'] = total_liabs

            total_equity = sheet['Share Capital & Additional Paid-In Capital_balance'] - sheet['Treasury Stock_balance'] + sheet['Retained Earnings_balance']
            sheet['Total Equity_balance'] = total_equity

            bal_check = curr_assets + noncurr_assets - curr_liabs - noncurr_liabs - total_equity

            sheet['BALANCE CHECK_balance'] = bal_check
            print(bal_check)

        return
    
    def dcf_calcs_sim(self, years, sheet, risk_free, beta, market_return, iterations=7000):
        '''calculate:
        - weighted avg debt
        - weighted avg equity
        - risk free rate
        - beta
        - expected growth rate (avg historical growth)
        - expected market return
        '''

        
        cur_year = years[0] 
        cur_year_int = cur_year - (years[0] + 1)
        final_year = years[-1] 
        fin_year_int = final_year - (years[0] + 1)
    

        wavg_debt = sheet[sheet['Fiscal Year_balance'] == cur_year]['Total Liabilities_balance'].iloc[0] / sheet[sheet['Fiscal Year_balance'] == cur_year]['Total Assets_balance'].iloc[0]
        wavg_equity = sheet[sheet['Fiscal Year_balance'] == cur_year]['Total Equity_balance'].iloc[0] / sheet[sheet['Fiscal Year_balance'] == cur_year]['Total Assets_balance'].iloc[0]
        
        simulation_lst = []
        
        for x in range(iterations):
            avg_growth = randint(2,4) / 100

            # net_debt = 

            cost_debt = sheet[sheet['Fiscal Year_balance'] == cur_year]['Interest Expense, Net_income'].iloc[0] / sheet[sheet['Fiscal Year_balance'] == cur_year]['Total Liabilities_balance'].iloc[0]
            # cost_debt = cost_debt * randint(75,125)/100

            cost_equity = risk_free + (beta*(market_return-risk_free))
            cost_equity = cost_equity * randint(75,125)/100

            wacc = (wavg_debt*cost_debt) + (wavg_equity*cost_equity)
            
            # Unlevered Free Cash Flow_cashflow
            final_year_fcf = sheet[sheet['Fiscal Year_balance'] == final_year]['Unlevered Free Cash Flow_cashflow'].iloc[0]
            final_year_fcf = final_year_fcf * randint(75,125)/100
            
            future_term_val = (final_year_fcf*(1+avg_growth))/(wacc-avg_growth)

            # print(wacc, avg_growth)

            final_year_disc = (1-wacc)**fin_year_int

            pv_tv = future_term_val*final_year_disc
            
            # disc factor dict?
            yrs = 0
            present_vals = 0
            for i in years:
                yrs += 1
                disc_factor = (1-wacc)**yrs

                pv_fcf = disc_factor * sheet[sheet['Fiscal Year_balance'] == i]['Unlevered Free Cash Flow_cashflow'].iloc[0] * randint(75,125)/100
                present_vals += pv_fcf
            
            ent_val = present_vals + pv_tv

            total_debt = sheet[sheet['Fiscal Year_balance'] == cur_year]['Total Liabilities_balance'].iloc[0]
            total_cash = sheet[sheet['Fiscal Year_balance'] == cur_year]['Cash, Cash Equivalents & Short Term Investments_balance'].iloc[0]
            equity_val = ent_val - total_debt + total_cash
            
            shares_out = sheet[sheet['Fiscal Year_balance'] == cur_year - 1]['Shares (Basic)_balance'].iloc[0] # assumes same shares out as year year of projections (first project year-1)
            # last_close = sheet[sheet['Fiscal Year_balance'] == cur_year - 1]['Close_income'].iloc[0]

            intrinsic_price = equity_val/shares_out
            print(intrinsic_price)
        
            simulation_lst.append(intrinsic_price)
            # calculate change of negativity

        # simulation_array = numpy.array(simulation_lst)
        return simulation_lst

        
        
        # terminal_val =  

        return

    def score(self):
        return


class AccDilution:
    def __init__(self, ticker, balance, income):
        # self.role = role
        self.ticker = ticker
        self.cur_yr = 2022
        self.purchase_premium = .25
        self.marginal_tax = .21

        self.interest_rate = 0
        
        self.transaction_fees = 0
        self.incremental_da = 0
        self.net_synergies = 0

        self.stock_consideration_pct = .5
        self.cash_consideration_pct = 1 - self.stock_consideration_pct
        # self.offer_val = None
        return

    def main(self):
        deal_info = self.deal_inputs()
        
        end_eps = []

        for res in deal_info:
            offer_value = res['target_share_price'] * res['target_shares_out'] * self.purchase_premium
            stock_consideration = offer_value*self.stock_consideration_pct
            shares_issued = stock_consideration / res['acq_share_price']

            consolidated_ebt = res['acq_net_income'] + res['target_net_income'] *(1+self.marginal_tax) # check if this is right rate to use or if should be w avg.
            proforma_ebt = consolidated_ebt - (self.interest_rate * self.transaction_fees) - (self.transaction_fees) + (self.net_synergies) - (self.incremental_da)
            proforma_net_income =  proforma_ebt*(1-self.marginal_tax)

            new_total_shares = shares_issued + res['acq_shares_out']
            proforma_eps = proforma_net_income / new_total_shares

            change = (proforma_eps- (res['acq_net_income']/res['acq_shares_out'])) / (res['acq_net_income']/res['acq_shares_out'])
            
            end_result = {}
            end_result['target'] = res['target_ticker']
            end_result['acquirer'] = res['acq_ticker']
            end_result['accretion-dilution'] = change

            end_eps.append(end_result)

        return pd.DataFrame(end_eps)

    def deal_inputs(self):
        shares_out = income[(income['Ticker'] == ticker) & (income['Fiscal Year'] == self.cur_yr)]['Shares (Basic)'].iloc[-1]
        net_income  = income[(income['Ticker'] == ticker) & (income['Fiscal Year'] == self.cur_yr)]['Net Income'].iloc[-1]
        
        yf_ticker = yf.Ticker(ticker)        
        price_df = yf_ticker.history(period='max')

        target_share_price = price_df['Close'].iloc[-1]
        results = []

        for x in set(income['Ticker'].unique()):
            acq_ticker = yf.Ticker(x)        
            acq_price_df = acq_ticker.history(period='max')
            
            res = {}
            
            try:
                res['acq_share_price'] = acq_price_df['Close'].iloc[-1]
                res['acq_shares_out'] = income[(income['Ticker'] == x) & (income['Fiscal Year'] == self.cur_yr)]['Shares (Basic)'].iloc[-1]
                res['acq_ticker'] = x
                res['acq_net_income'] = income[(income['Ticker'] == x) & (income['Fiscal Year'] == self.cur_yr)]['Net Income'].iloc[-1]

                res['target_share_price'] = target_share_price
                res['target_shares_out'] = shares_out
                res['target_ticker'] = ticker
                res['target_net_income'] = net_income
                
                res['purchase_price'] = res['target_share_price']*self.purchase_premium
                
                if res['acq_share_price']*res['acq_shares_out'] > target_share_price * shares_out:
                    results.append(res)

            except: 
                pass
                
        return results

    def score(self):
        return

# class FinalSiteTables:
    

ticker = 'FLWS'
t = Tables(ticker)

print(t.str_industry)

financials = t.get_financials()
print(financials.keys())


# ##### Important for all functions (e.g., DCF)
balance = financials['balance']
income = financials['income']


get_tables = input('Get tables? (Y / N)?: ')

if get_tables == 'Y':
    derived = financials['derived']
    cashflow = financials['cashflow']
    ratios = financials['ratios']

    price_rats = t.price_ratios()
    # price_rats.to_csv('{}/priceratios.csv'.format(ticker))
    returns = t.get_returns()
    horizon_score = t.horizon_score(ratios=ratios)
    
    tecnhicals = t.get_technicals().iloc[:, -252*2:]

    ec = t.economics()

    ad = AccDilution(ticker, balance, income)
    ad_result = ad.main()

    writer = pd.ExcelWriter("{}/{}-data.xlsx".format(ticker, ticker), engine="xlsxwriter")

    # Write each dataframe to a different worksheet.
    cashflow.to_excel(writer, sheet_name='cashflow')
    balance.to_excel(writer, sheet_name='balance')
    income.to_excel(writer, sheet_name="income")
    derived.to_excel(writer, sheet_name='simfin_derived')
    ratios.to_excel(writer, sheet_name="ratios")
    price_rats.to_excel(writer, sheet_name="market_ratios")
    returns.to_excel(writer, sheet_name="returns")
    horizon_score.to_excel(writer, sheet_name="horizon_scores")
    tecnhicals.to_excel(writer, sheet_name="tecnicals")
    ad_result.to_excel(writer, sheet_name="accretion_dilution")
    ec.to_excel(writer, sheet_name="economics")
    
    # # Close the Pandas Excel writer and output the Excel file.
    writer.close()

#####

dcf = DCF(balance, income, ticker)

projections_done = input('Projections Done? (Y / N / Project): ')

if projections_done == 'Y':
    sheet = pd.read_csv('{}/dcf-projections.csv'.format(ticker))
    projections = dcf.project_inputs(sheet)
    projections.to_csv('{}/dcf-projections-final.csv'.format(ticker))
elif projections_done == 'N':
    print('Please project inputs using this sheet {}/dcf-projections.csv'.format(ticker))
elif projections_done == 'Project':
    projections = dcf.create_projections()
    projections.to_csv('{}/dcf-projections.csv'.format(ticker))



# https://stackoverflow.com/questions/42092263/combine-multiple-csv-files-into-a-single-xls-workbook-python-3

# ticker = t.ticker

# dcf = DCF(financials['balance'], financials['income'], ticker) # add api key arg and move class elsewhere
# dcf.create_projections()
#dcf.project_inputs()
# dcf.simulate()
