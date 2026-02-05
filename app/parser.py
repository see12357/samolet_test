import json
import asyncio
from playwright.async_api import async_playwright
from typing import Dict, Any, List, Union


class SamoletParser:
    def __init__(self):
        self.user_agent = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"

    async def get_apartment_data(self, url: str) -> Dict[str, Any]:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled"]
            )
            context = await browser.new_context(user_agent=self.user_agent)
            page = await context.new_page()

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=45000)

                await asyncio.sleep(3)

                try:
                    await page.wait_for_selector("script#__NUXT_DATA__", timeout=10000)
                    raw_json_str = await page.locator("script#__NUXT_DATA__").inner_text()
                except Exception:
                    try:
                        raw_json_str = await page.locator("script#__NEXT_DATA__").inner_text()
                    except:
                        raise ValueError("Не удалось найти данные на странице (защита или неверная ссылка).")

                data_list = json.loads(raw_json_str)

                flat_root_index = None

                if isinstance(data_list, list):
                    for item in data_list:
                        if isinstance(item, dict):
                            for k, v in item.items():
                                if k.startswith('getFlatDetail'):
                                    flat_root_index = v
                                    break
                        if flat_root_index is not None:
                            break

                    if flat_root_index is None:
                        raise ValueError("Индекс квартиры не найден в структуре.")

                    flat_data_raw = self._resolve(data_list, flat_root_index)
                else:
                    flat_data_raw = data_list.get('props', {}).get('pageProps', {}).get('flat', {})

                if not flat_data_raw or not isinstance(flat_data_raw, dict):
                    raise ValueError("Пустые данные квартиры.")

                project_data = self._resolve(data_list, flat_data_raw.get('project')) if isinstance(data_list,
                                                                                                    list) else flat_data_raw.get(
                    'project', {})

                result = {
                    "District": project_data.get('title', 'Unknown') if isinstance(project_data, dict) else "Unknown",
                    "TotalArea": float(self._resolve(data_list, flat_data_raw.get('area', 0))),
                    "KitchenArea": float(self._resolve(data_list, flat_data_raw.get('kitchen_area', 0))),
                    "Floor": int(self._resolve(data_list, flat_data_raw.get('floor', 1))),
                    "FloorsTotal": int(self._resolve(data_list, flat_data_raw.get('total_floors', 1))),
                    "CeilingHeight": float(self._resolve(data_list, flat_data_raw.get('ceiling_height', 2.7))),
                    "PropertyType": f"{self._resolve(data_list, flat_data_raw.get('rooms', 1))} ккв",
                    "Finishing": "Чистовая" if self._resolve(data_list,
                                                             flat_data_raw.get('is_finish_done')) else "Без отделки",
                }
                return result

            except Exception as e:
                raise ValueError(f"Playwright Error: {str(e)}")
            finally:
                await browser.close()

    def _resolve(self, data_list: Any, value: Any) -> Any:
        """Вспомогательная функция для Nuxt"""
        if isinstance(data_list, list) and isinstance(value, int):
            if 0 <= value < len(data_list):
                candidate = data_list[value]
                if isinstance(candidate, (dict, list, str)):
                    return candidate
        return value