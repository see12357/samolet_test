from langchain_core.tools import tool
from app.parser import SamoletParser
from app.preprocessor import InferencePreprocessor
from app.engine import ValuationEngine

preproc = InferencePreprocessor()
engine = ValuationEngine()
samolet_parser = SamoletParser()

@tool
def evaluate_manual(
    district: str,
    area: float,
    floor: int,
    floors_total: int,
    rooms: int = 1,
    ceiling_height: float = 2.7,
    finishing: str = "Без отделки",
    property_category: str = "Квартира",
    building_type: str = "Монолит"
) -> str:
    """
    Calculates price based on detailed manual parameters.
    Args:
        district: Location or complex name.
        area: Total area in m2.
        floor: Current floor.
        floors_total: Total floors.
        rooms: Number of rooms.
        ceiling_height: Height of ceilings (e.g. 2.9).
        finishing: Renovation type ('Чистовая', 'Подчистовая', 'Без отделки').
        property_category: 'Квартира' or 'Апартамент'.
        building_type: Material like 'Монолит' or 'Панель'.
    """
    raw = {
        "District": district,
        "TotalArea": area,
        "Floor": floor,
        "FloorsTotal": floors_total,
        "rooms_count": rooms,
        "CeilingHeight": ceiling_height,
        "Finishing": finishing,
        "PropertyCategory": property_category,
        "BuildingType": building_type,
        "PropertyType": f"{rooms} ккв"
    }
    res = engine.predict(preproc.process(raw))
    impacts_str = ", ".join([f"{i['feature']} ({i['value']*100:+.1f}%)" for i in res["impacts"]])
    return f"ОЦЕНКА: {res['total_price']:,} руб. ({res['price_m2']:,} р/м2). Факторы: {impacts_str}"

@tool
async def evaluate_by_url(url: str) -> str:
    """Extracts data from Samolet.ru and calculates price."""
    try:
        raw_data = await samolet_parser.get_apartment_data(url)
        res = engine.predict(preproc.process(raw_data))
        impacts_str = ", ".join([f"{i['feature']} ({i['value']*100:+.1f}%)" for i in res["impacts"]])
        return f"ОБЪЕКТ: {raw_data['District']}, {raw_data['TotalArea']}м2. ОЦЕНКА: {res['total_price']:,} руб. Факторы: {impacts_str}"
    except Exception as e:
        return f"Ошибка парсинга: {e}. Введите данные вручную."