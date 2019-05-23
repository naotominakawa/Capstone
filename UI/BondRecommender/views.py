import os
import json
from functools import lru_cache
from urllib.parse import urlencode

import pandas as pd
from django.http import HttpResponse
from django.shortcuts import render, redirect
from django.urls import reverse

from BondRecommender.models import Securities, vanguard_merge
from BondRecommender.data_loader import get_single_day_data, get_multi_day_data
from BondRecommender.recommendation_models import similar_bonds_pipeline
from BondRecommender.prediction_models import predict_single_rc
from BondRecommender.bpr_model import BPRModel, ModelHelper

@lru_cache(maxsize=16)
def get_bpr_model_helper():
    num_factors = 32
    
    isin_to_index_mapping_file = os.path.join(os.path.dirname(__file__), 'isin_to_index2.json')
    isin_to_index_mapping = json.load(open(isin_to_index_mapping_file))
    num_bonds = len(isin_to_index_mapping)
    
    model = BPRModel(num_bonds, num_factors)
    model_helper = ModelHelper(model, isin_to_index_mapping, get_single_day_data())
    
    weights_file = os.path.join(os.path.dirname(__file__), 'bpr_v2.pt')
    model_helper.load(weights_file)

    return model_helper 


# Create your views here.
def home(request):
    context = dict()
    return render(request, 'home.html', context)

def results(request):

    context = dict()
    sec_id = str(request.GET['sec_id'])
    context['sec_id'] = sec_id
    context['sec_id_error'] = 0
    numofdays = int(request.GET.get('num_of_days', 30))
    numofrecomm = int(request.GET.get('num_of_recommendation', 10))
    cohort_filtering = str(request.GET.get('cohort_filtering', 'Yes'))
    ytm_upper = str(request.GET.get('ytm_upper', ''))
    ytm_lower = str(request.GET.get('ytm_lower', ''))
    oad_upper = str(request.GET.get('oad_upper', ''))
    oad_lower = str(request.GET.get('oad_lower', ''))

    try:
        ## COLUMNS WHOSE VALUES MUST BE THE SAME IN ORDER TO BE CONSIDERED SIMILAR
        if cohort_filtering == "Yes":
            cohort_attributes = ['BCLASS3',  'Ticker', 'Country', 'Class - Detail - Code']
        else:
            cohort_attributes = None

        ## COLUMNS THAT THE MODEL SHOULD CONSIDER WHEN LOOKING FOR SIMILAR BONDS
        features = ["OAS", "OAD", "KRD 5Y", "KRD 10Y", "KRD 20Y", "KRD 30Y"]

        ## COLUMNS TO DISPLAY IN THE CLI OUTPUT
        display_columns = ['ISIN', 'Ticker', 'BCLASS3', 'Country'] + (features or []) + ['Yield to Mat', 'Cpn', 'Px Close']
        #context['display_columns'] = display_columns

        bond = get_single_day_data().get_bond(sec_id)

        filter_conditions = {'Yield to Mat': (ytm_lower, ytm_upper), 'OAD': (oad_lower, oad_upper)}

        bond = bond.reset_index()
        bond_table = bond[display_columns]
        context['bond_table'] = bond_table.values.tolist()

        model_helper = get_bpr_model_helper()

        similar_bonds = model_helper.predict(sec_id)
        similar_bonds = model_helper.display(similar_bonds, display_cols=display_columns).reset_index()
        similar_bonds_table = similar_bonds.values.tolist()

        bond = get_multi_day_data(numdays=numofdays).get_bond(sec_id)
        context['bond_dates'] = '|'.join(bond["date"].values)
        context['bond_OAS'] = '|'.join(list(map(str, bond["OAS"].values)))

        bond_dates2 = list()
        bond_OAS2 = list()

        for l in similar_bonds_table:
            sec_code = str(l[1])
            bond = get_multi_day_data(numdays=numofdays).get_bond(sec_code)
            bond_dates2.append([sec_code, '|'.join(bond["date"].values)])
            bond_OAS2.append([sec_code, '|'.join(list(map(str, bond["OAS"].values)))])

        context['bond_dates2'] = bond_dates2
        context['bond_OAS2'] = bond_OAS2

        # Integration with JPM rich/cheap
        isins = similar_bonds.ISIN.unique()
        prediction_result = pd.concat([predict_single_rc(date=None, isin=isin) for isin in isins])
        result = similar_bonds.merge(prediction_result, on="ISIN", how ="outer")
        display_columns = display_columns + ["rich/cheap"]
        result_table = result[display_columns].values.tolist()
        context["similar_bonds_table"] = result_table
        context['display_columns'] = display_columns

        return render(request, 'results.html', context)

    except KeyError as e:
        import traceback
        print(traceback.print_exc())
        context['sec_id_error'] = 1
        return render(request, 'home.html', context)


def feedback(request):
    """
    Update the model with feedback from the user, and refresh the predictions
    :param request:     The RequestContext, expected to have three attributes of the GET request
                            bond:   The ISIN of the bond we are finding bonds similar to
                            better: The ISIN of a bond that is a better recommendation than its current rank
                            worse:  The ISIN of a bond that is a worse recommendation than its current rank
    """

    bond = request.GET['bond']
    better = request.GET['better']
    worse = request.GET['worse']

    feedback = [(bond, better, worse)]
    get_bpr_model_helper().process_feedback(feedback)
    # TODO add support for persisting the feedback and/or the model
    # model_helper.save('models/bpr_v3.pt')

    base_url = reverse('results')
    query_string = urlencode({'sec_id': bond})
    url = "{}?{}".format(base_url, query_string)

    return redirect(url)


def get_similar_bonds(isin, single_day_data, num_of_bonds, filter_conditions, features=None, cohort_attributes=None):
    """
    This is a top-level function that is meant to be called when processing a server requst for SimilarBonds

    :param isin:                The ISIN identifier of the bond we're trying to find similar bonds for
    :param features:            Optional. A list of columns to consider when determining bond similarity.
                                Default: None, meaning all columns in the data set
    :param cohort_attributes:   Optional. A list of columns specifying the bond attributes that *must* be the same in order for a bond to be considered similar
                                Default: None, meaning all bonds are valid candidates
    """

    # ISIN to Pandas Series of data for that bond
    bond = get_single_day_data().get_bond(isin)

    # Pandas Series to Pandas DataFrame of data for all bonds in the specified cohort
    if cohort_attributes is None:
        cohort_attributes = []
        bond_cohort = get_single_day_data().data
    else:
        bond_cohort = get_single_day_data().get_cohort(bond, attributes=cohort_attributes)

    ytm_lower = filter_conditions['Yield to Mat'][0]
    ytm_upper = filter_conditions['Yield to Mat'][1]
    oad_lower = filter_conditions['OAD'][0]
    oad_upper = filter_conditions['OAD'][1]

    # Filter by ytm range
    if (ytm_upper != '') and (ytm_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'].between(float(ytm_lower), float(ytm_upper), inclusive=True)]
    elif (ytm_upper != '') and (ytm_lower == ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] < float(ytm_upper)]
    elif (ytm_upper == '') and (ytm_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] > float(ytm_lower)]
    else:
        pass

    # Filter by oad range
    if (oad_upper != '') and (oad_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['OAD'].between(float(oad_lower), float(oad_upper), inclusive=True)]
    elif (oad_upper != '') and (oad_lower == ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] < float(oad_upper)]
    elif (oad_upper == '') and (oad_lower != ''):
        bond_cohort = bond_cohort[bond_cohort['Yield to Mat'] > float(oad_lower)]
    else:
        pass

    if features is None:
        features = [col for col in bond_cohort.columns if col not in cohort_attributes]

    # Fit the model
    model = similar_bonds_pipeline()
    model.fit(bond_cohort[features])

    # Find similar bonds
    k_neighbors = min(bond_cohort.shape[0], num_of_bonds)
    distances, indices = model.predict(bond[features], k_neighbors=k_neighbors)
    similar_bond_isins = bond_cohort.iloc[indices.ravel()].index.values
    # Exclude the input isin from the list of similar bonds
    similar_bond_isins = [i for i in similar_bond_isins if i != isin]

    similar_bonds = get_single_day_data().get_bonds(similar_bond_isins)

    return similar_bonds
