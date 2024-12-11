import json
from dataclasses import dataclass
from typing import Dict, List
from collections import Counter, defaultdict
import numpy as np
import pandas as pd


@dataclass
class Product:
    title: str
    modelID: str
    shop: str
    url: str
    featuresMap: Dict[str, str]

    def get_shop(self):
        return self.shop

def load_data():
    jsondata: Dict[str, List[Dict]]
    with open("TVs-all-merged.json") as f:
        jsondata = json.load(f)

    products: List[Product] = []

    for duplicates in jsondata.values():
        for product in duplicates:
            products.append(Product(title=product['title'].lower(),
                                    modelID=product['modelID'],
                                    shop=product['shop'],
                                    url=product['url'],
                                    featuresMap=product['featuresMap']))

    N = len(products)
    duplicates_matrix = np.zeros((N, N)).astype(int)

    for i, p1 in enumerate(products):
        for j, p2 in enumerate(products):
            if i != j and p1.modelID == p2.modelID:
                duplicates_matrix[i, j] = 1

    print(f'\nN={N} (of which {len(jsondata)} unique)\n')

    return products, duplicates_matrix


def main():
    products, duplicates_matrix = load_data()

    all_keys = []
    for product in products:
        all_keys.extend(product.featuresMap.keys())

    key_counts = Counter(all_keys)

    sorted_key_counts = key_counts.most_common()
    print("\nFeature Key Occurrences (Descending):\n")
    for key, count in sorted_key_counts:
        print(f"{key}: {count}")

    shop_keys = defaultdict(list)
    for product in products:
        shop_keys[product.shop].extend(product.featuresMap.keys())

    # Initialize a defaultdict for counting keys per shop
    shop_key_counts = defaultdict(lambda: defaultdict(int))

    # Count occurrences of each key in each shop
    for product in products:
        shop = product.shop
        for key in product.featuresMap.keys():
            shop_key_counts[key][shop] += 1

    # Extract all unique shops and keys
    all_shops = sorted({product.shop for product in products})
    all_keys = sorted(shop_key_counts.keys())

    # Create a matrix with keys as rows and shops as columns
    matrix_data = []
    for key in all_keys:
        row = [shop_key_counts[key].get(shop, 0) for shop in all_shops]
        matrix_data.append(row)

    # Create a DataFrame for better visualization
    matrix_df = pd.DataFrame(matrix_data, index=all_keys, columns=all_shops)
    filtered_matrix_df = matrix_df[(matrix_df > 0).sum(axis=1) >= 2]
    # Print the matrix
    print("\nMatrix of Key Counts Per Shop:\n")
    print(filtered_matrix_df)




if __name__ == '__main__':
    main()
