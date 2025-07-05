"""
Select business attributes for downstream modeling.

Selected attributes and their justification (based on EDA):

| Attribute                | Distribution                        | Justification                                                        |
|--------------------------|-------------------------------------|-----------------------------------------------------------------------|
| RestaurantsPriceRange2   | 2★(55%), 1★(40%), 3★(4%), 4★(0.5%) | High entropy, critical for filtering, stable indicator                |
| OutdoorSeating           | False(55%), True(45%)               | Good balance, significant for experience, enables contextual recs     |
| Ambience                 | Multiple values, rich diversity      | Captures experience, semantic richness, nuanced matching              |
| RestaurantsReservations  | False(62%), True(38%)               | Good entropy, reflects service, correlates with popularity/planning   |

This script cleans the 'attributes' field within each business entry, keeping only the attributes listed above.
The full business entry structure is preserved for downstream use.
"""
import json

def run(businesses, config):
    """
    Cleans the 'attributes' field for each business in-memory.
    Args:
        businesses (list): A list of business dictionaries.
        config (dict): The configuration dictionary.
    Returns:
        list: The list of business dictionaries with cleaned attributes.
    """
    selected_attrs = [
    "RestaurantsPriceRange2",
    "OutdoorSeating",
    "Ambience",
    "RestaurantsReservations"
]

    for business in businesses:
        raw_attrs = business.get('attributes')
        if not raw_attrs:
            business['attributes'] = {}
            continue

        attrs_dict = {}
        if isinstance(raw_attrs, str):
            try:
                attrs_dict = json.loads(raw_attrs)
            except Exception:
                import ast
                try:
                    attrs_dict = ast.literal_eval(raw_attrs)
                except Exception:
                    pass
        elif isinstance(raw_attrs, dict):
            attrs_dict = raw_attrs

        cleaned_attrs = {key: attrs_dict[key] for key in selected_attrs if key in attrs_dict}
        
        business['attributes'] = cleaned_attrs
    
    print("In-memory attribute cleaning complete.")
    return businesses

if __name__ == '__main__':
    main() 