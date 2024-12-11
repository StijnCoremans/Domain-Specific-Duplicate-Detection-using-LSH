import json
from dataclasses import dataclass
from typing import Dict, List, Set
import numpy as np
import pandas as pd
import re
import math
from tqdm import tqdm
import random
from encoder import ModelWordsEncoder, ShingleEncoder, DomainFeaturesEncoder, PlusFeaturesEncoder
from numpy.typing import NDArray
from lsh import lsh
from minhashing import minhash
from sklearn.cluster import AgglomerativeClustering

@dataclass
class Product:
    title: str
    modelID: str
    shop: str
    url: str
    featuresMap: Dict[str, str]
    domainFeatures : Dict[str, List[int]]

    def get_shop(self):
        return self.shop


def load_data():
    jsondata: Dict[str, List[Dict]]
    with open("TVs-all-merged.json") as data:
        jsondata = json.load(data)

    products = []

    for tv_list in jsondata.values():
        for product in tv_list:

            domainFeatures = domain_features(product['featuresMap'])

            products.append(Product(title=product['title'].lower(),
                                    modelID=product['modelID'],
                                    shop=product['shop'],
                                    url=product['url'],
                                    featuresMap=product['featuresMap'],
                                    domainFeatures=domainFeatures))
            print(domainFeatures)
    N = len(products)
    duplicates_matrix = np.zeros((N, N)).astype(int)

    for i, product_i in enumerate(products):
        for j, product_j in enumerate(products):
            if i != j and product_i.modelID == product_j.modelID:
                duplicates_matrix[i, j] = 1

    print(f'\nN={N} (of which {len(jsondata)} unique)\n')

    return products, duplicates_matrix


def domain_features(featuresMap: Dict[str, str]):
    
    keys = ["Maximum Resolution", "Aspect Ratio", "Brand", "Screen Size", "VESA", "Height", "Depth", "Width", "Dimension", "Weight", "Refresh Rate", "Energy Consumption", "HDMI"]
    # ,"Brightness","Dynamic Contrast Ratio"

    keys_lower = [key.lower() for key in keys]

    matched_features = {}
    num_pattern = r"[-+]?\d*\.\d+|\d+"

    for feature_key, feature_value in featuresMap.items():

        feature_key_lower = feature_key.lower()

        for key in keys_lower:
            if key in feature_key_lower:

                if key == "brand" :
                    if key in matched_features:
                        matched_features[key].append(feature_value.lower())
                    else:
                        matched_features[key] = [feature_value]
                else:

                    numbers = re.findall(num_pattern, feature_value)
                    numbers = [float(num) if '.' in num else int(num) for num in numbers]

                    if key in matched_features:
                        matched_features[key].extend(numbers)
                    else:
                        matched_features[key] = numbers


    return matched_features
        

def diff_brand(a: Product, b: Product):
    
    brands = ["akai", "alba", "apple", "arcam", "arise", "bang", "bpl", "bush", "cge", "changhong", "compal", "curtis",
          "durabrand", "element", "finlux", "fujitsu", "funai", "google", "haier", "hisense", "hitachi", "itel",
          "jensen", "jvc", "kogan", "konka", "lg", "loewe", "magnavox", "marantz", "memorex", "micromax", "metz",
          "onida", "panasonic", "pensonic", "philips", "planar", "proscan", "rediffusion", "saba", "salora", "samsung",
          "sansui", "sanyo", "seiki", "sharp", "skyworth", "sony", "tatung", "tcl", "telefunken", "thomson", "toshiba",
          "tpv", "tp vision", "vestel", "videocon", "vizio", "vu", "walton", "westinghouse", "xiaomi", "zenith"]

    a_str = a.title + ' ' + ' '.join(a.featuresMap.values()).lower()
    b_str = b.title + ' ' + ' '.join(b.featuresMap.values()).lower()

    for brand in brands:
        a_has_brand = brand in a_str
        b_has_brand = brand in b_str

        if a_has_brand != b_has_brand:
            return True

    return False

def Ddiff_brand(a: Product, b: Product):
    
    brands = ["akai", "alba", "apple", "arcam", "arise", "bang", "bpl", "bush", "cge", "changhong", "compal", "curtis",
          "durabrand", "element", "finlux", "fujitsu", "funai", "google", "haier", "hisense", "hitachi", "itel",
          "jensen", "jvc", "kogan", "konka", "lg", "loewe", "magnavox", "marantz", "memorex", "micromax", "metz",
          "onida", "panasonic", "pensonic", "philips", "planar", "proscan", "rediffusion", "saba", "salora", "samsung",
          "sansui", "sanyo", "seiki", "sharp", "skyworth", "sony", "tatung", "tcl", "telefunken", "thomson", "toshiba",
          "tpv", "tp vision", "vestel", "videocon", "vizio", "vu", "walton", "westinghouse", "xiaomi", "zenith"]

    a_brand = a.domainFeatures.get("brand")
    b_brand = b.domainFeatures.get("brand")

    if a_brand == None or b_brand == None:
        
        a_title = a.title.lower()
        b_title = b.title.lower()

        for brand in brands:
            a_has_brand = brand in a_title
            b_has_brand = brand in b_title

            if a_has_brand != b_has_brand:
                return True

        return False

    for brand in brands:
        brand_a = any(brand in item for item in a_brand)
        brand_b = any(brand in item for item in b_brand)

        if brand_a != brand_b:
            return True

    return False


def preprocess(products: List[Product]):
    
    mw_encoder = ModelWordsEncoder()
    df_encoder = DomainFeaturesEncoder()

    mw = [mw_encoder.encode(product.title.lower()) for product in products]
    df = [df_encoder.encode(product.domainFeatures) for product in products]

    combined_matrix = [mw[i].union(df[i]) for i in range(len(products))]
    
    return mw_encoder, df_encoder, combined_matrix

def preprocess_MSMP_plus(products: List[Product]):
    
    mw_encoder = ModelWordsEncoder()
    plus_encoder = PlusFeaturesEncoder()

    mw = [mw_encoder.encode(product.title.lower()) for product in products]
    plus = [plus_encoder.encode(product.featuresMap) for product in products]

    combined_matrix = [mw[i].union(plus[i]) for i in range(len(products))]
    
    return mw_encoder, plus_encoder, combined_matrix

def preprocess_MSMP(products: List[Product]):
    
    mw_encoder_MSMP = ModelWordsEncoder()

    combined_matrix = [mw_encoder_MSMP.encode(product.title.lower()) for product in products]
    
    return mw_encoder_MSMP,  combined_matrix

def apply_clustering(products: List[Product],
                     candidate_pairs: NDArray,
                     k: int,
                     mu=.650,
                     gamma=.756,
                     distance_threshold=.522):
    N = len(products)
    inf_distance = 1000
    distances = np.ones((N, N)) * inf_distance
    np.fill_diagonal(distances, 0)

    def set_distance(i: int, j: int, value):
        distances[i, j] = value
        distances[j, i] = value

    for i, j in tqdm(candidate_pairs, desc="Clustering", leave=False):
        if i == j:
            continue

        product_i: Product = products[i]
        product_j: Product = products[j]

        if product_i.shop == product_j.shop or diff_brand(product_i, product_j):
            set_distance(i, j, inf_distance)
            continue

        strict_domain_blocking = False # True for strict domain blocking 
        if strict_domain_blocking:
            amount_corresponding_features = 0
            amount_similar_features = 0

            for domain_feature_i_key, domain_feature_i_value in product_i.domainFeatures.items():
                for domain_feature_j_key, domain_feature_j_value in product_j.domainFeatures.items():
                    if domain_feature_i_key == domain_feature_j_key:
                        amount_corresponding_features += 1
                        for value in domain_feature_i_value:
                            if value in domain_feature_j_value:
                                amount_similar_features += 1
                                break

            if amount_corresponding_features < 2:
                amount_corresponding_features = amount_corresponding_features
            elif amount_similar_features < 0.5 * amount_corresponding_features:
                set_distance(i, j, inf_distance)
                continue


        sim = 0
        m = 0
        w = 0

        nonmatching_i: Dict[str, str] = dict(product_i.featuresMap)
        nonmatching_j: Dict[str, str] = dict(product_j.featuresMap)

        for q_key, q_value in product_i.featuresMap.items():
            for r_key, r_value in product_j.featuresMap.items():
                key_similarity = overlapping_q_grams(ShingleEncoder.shingle(q_key, k),
                                         ShingleEncoder.shingle(r_key, k))

                if key_similarity > gamma:
                    value_similarity = overlapping_q_grams(ShingleEncoder.shingle(q_value, k),
                                               ShingleEncoder.shingle(r_value, k))
                    weight = key_similarity
                    sim += weight * value_similarity
                    m += 1
                    w += weight
                    
                    value = nonmatching_i.get(q_key, None)
                    if value is not None:
                        nonmatching_i.pop(q_key)
                    
                    value2 = nonmatching_j.get(r_key, None)
                    if value2 is not None:
                        nonmatching_j.pop(r_key)

        avg_sim = sim / w if w > 0 else 0
        mw_perc = percentage_match(extract_model_words(nonmatching_i), extract_model_words(nonmatching_j))
        title_sim = title_similarity(product_i.title.lower(), product_j.title.lower())

        min_features = min(len(product_i.featuresMap), len(product_j.featuresMap))
        h_sim: float
        if title_sim == -1:
            theta1 = m / min_features
            theta2 = 1 - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_perc
        else:
            theta1 = (1 - mu) * m / min_features
            theta2 = 1 - mu - theta1
            h_sim = theta1 * avg_sim + theta2 * mw_perc + mu * title_sim

        set_distance(i, j, 1 - h_sim)

    model = AgglomerativeClustering(distance_threshold=distance_threshold,
                                    n_clusters=None,
                                    linkage='single',
                                    metric='precomputed')
    model.fit(distances)

    comparisons_made = np.zeros((N, N))

    for i, j in np.argwhere((distances > 0) & (distances < inf_distance)):
        comparisons_made[min(i, j), max(i, j)] = 1

    num_comparisons_made = comparisons_made.sum()

    return model, num_comparisons_made


def overlapping_q_grams(value1: Set[str], value2: Set[str]):
    if len(value1) == 0 and len(value2) == 0:
        return 0

    n_1 = len(value1)
    n_2 = len(value2)

    distance = len(value1.symmetric_difference(value2))

    return (n_1 + n_2 - distance) / (n_1 + n_2)

def percentage_match(value1: Set[str], value2: Set[str]):
    if len(value1) == 0 or len(value2) == 0:
        return 0

    n_intersect = len(value1.intersection(value2))
    n_union = len(value1.union(value2))

    return n_intersect / n_union

def title_similarity(title1: str, title2: str):
    if len(title1) == 0 or len(title2) == 0:
        return -1

    alpha = .602

    if (cosine_similarity(clean_and_split(title1), clean_and_split(title2)) > alpha):
        return 1

    beta = 0
    sim = cosine_similarity(set(ModelWordsEncoder.get_model_words(title1)), set(ModelWordsEncoder.get_model_words(title2)))

    if (sim > beta):
        return sim

    return -1

def clean_and_split(title: str) -> set:
    
    noise_words = {"and", "or"}
    noise_chars = r"[&/\-]" 
    cleaned_title = re.sub(noise_chars, " ", title)

    words = cleaned_title.lower().split()
    filtered_words = {word for word in words if word not in noise_words}

    return filtered_words

def cosine_similarity(value1: Set[str], value2: Set[str]):
    if len(value1) == 0 and len(value2) == 0:
        return 0

    n_intersect = len(value1.intersection(value2))
    return n_intersect / (math.sqrt(len(value1)) + math.sqrt(len(value2)))

def performance_metrics(prefix: str,
                        predicted_duplicates: NDArray,
                        actual_duplicates: NDArray,
                        num_comparisons_made: int):
    duplicates_found = np.sum(predicted_duplicates * actual_duplicates)/ 2
    total_num_duplicates = np.sum(actual_duplicates) / 2

    pair_quality = duplicates_found / num_comparisons_made
    pair_completeness = duplicates_found / total_num_duplicates
    f1 = (2 * pair_quality * pair_completeness) / (pair_quality + pair_completeness)

    return {
        f'{prefix}f1': f1,
        f'{prefix}PQ': pair_quality,
        f'{prefix}PC': pair_completeness,
        f'{prefix}D_f': duplicates_found,
        f'{prefix}N_c': num_comparisons_made,
        f'{prefix}D_n': total_num_duplicates,
    }

def extract_model_words(attributes: Dict[str, str]) -> Set[str]:
    result: Set[str] = set()

    for value in attributes.values():
        for mw in ModelWordsEncoder.get_model_words(value):
            result.add(mw)

    return result

def main():
    products, duplicates_matrix = load_data()
    N = 1000

    replications = 1
    n = 500
    k = 3

    index_range = range(len(products))
    experiment_results: List[Dict] = []

    for bootstrap in tqdm(range(replications), desc="Replications"):
        current_indices = random.sample(index_range, k=N)
        current_products = [products[i] for i in current_indices]
        current_duplicates = np.array(
            [[duplicates_matrix[i, j] for j in current_indices] for i in current_indices])
        current_num_duplicates = np.sum(current_duplicates) / 2

        #mw_encoder, df_encoder, combined_matrix = preprocess(current_products)
        #mw_encoder_MSMP, combined_matrix_MSMP = preprocess_MSMP(current_products) # no domain features case: update matrix m = 
        mw_encoder_MSMP_plus, plus_encoder, combined_matrix_MSMP_plus = preprocess_MSMP_plus(current_products) # MSMP+ update matrix m = 

        for r in tqdm([r for r in range(2, n) if n % r == 0], desc="(r,b) combinations", leave=False):
            b = round(n / r)

            m = minhash(combined_matrix_MSMP_plus, n=r * b, vector_length = mw_encoder_MSMP_plus.vocabulary_size() + plus_encoder.vocabulary_size())

            if np.isinf(m).sum() > 0:
                raise ValueError(f"M still contains infinite values")

            candidate_pairs = lsh(m, b, r)
            lsh_predicted_duplicates = np.zeros_like(current_duplicates)

            for i, j in candidate_pairs:
                lsh_predicted_duplicates[i, j] = 1
                lsh_predicted_duplicates[j, i] = 1

            model, num_comparisons_made = apply_clustering(current_products, candidate_pairs, k=k)
            predicted_duplicates = np.array(
                [[int(model.labels_[i] == model.labels_[j]) for j in range(N)] for i in range(N)])

            experiment_results.append({
                'bootstrap': bootstrap,
                'n': n,
                'b': b,
                'r': r,
                'num_duplicates': current_num_duplicates,
                **performance_metrics('lsh__',
                                      lsh_predicted_duplicates,
                                      current_duplicates,
                                      candidate_pairs.shape[0]),
                **performance_metrics('clu__',
                                      predicted_duplicates,
                                      current_duplicates,
                                      num_comparisons_made)
            })
            df = pd.DataFrame(experiment_results).astype(float)
            print(df)

    df_results = pd.DataFrame(experiment_results).astype(float)
    print(df_results)
    df_results.to_csv("results_all_MSM.csv")

    def get_best_for_bootstrap(i: int, key: str):
        filtered_df = df_results[df_results['bootstrap'] == i]
        f1 = filtered_df[key]
        return filtered_df[f1 == f1.max()]

    best_f1_star = pd.concat([get_best_for_bootstrap(i, 'lsh__f1') for i in range(replications)])
    print('\n\nBest F1-star results per bootstrap:\n')
    print(
        best_f1_star[['bootstrap', 'n', 'b', 'r', 'lsh__f1', 'lsh__PQ', 'lsh__PC', 'lsh__D_f', 'lsh__N_c', 'lsh__D_n']])
    best_f1_star.to_csv("results_best_f1_star_MSMP_plus.csv")

    best_f1 = pd.concat([get_best_for_bootstrap(i, 'clu__f1') for i in range(replications)])
    print('\n\nBest F1 results per bootstrap:\n')
    print(best_f1[['bootstrap', 'n', 'b', 'r', 'clu__f1', 'clu__PQ', 'clu__PC', 'clu__D_f',
                   'clu__N_c', 'clu__D_n']])
    best_f1.to_csv("results_best_f1_MSM.csv")



if __name__ == '__main__':
    main()

