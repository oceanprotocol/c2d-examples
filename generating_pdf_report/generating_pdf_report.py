import matplotlib.pyplot as plt
import os
import json

from reportlab.lib.pagesizes import letter
from reportlab.platypus import Table
from reportlab.lib import colors
from reportlab.pdfgen import canvas


def get_filename():
    dids = os.getenv("DIDS", None)

    if not dids:
        print("No DIDs found in environment. Aborting.")
        return

    dids = json.loads(dids)

    for did in dids:
        filename = f"data/inputs/{did}/0"  # 0 for metadata service
        print(f"Reading asset file {filename}.")

        return filename

def extract_results(filename):
    with open(file=filename, mode='r') as stock_file:
        stock_data = json.load(stock_file)

        return stock_data["results"]

def get_top_10_closing_prices(results):
    def sort_func(e):
        return e['price']
    
    return sorted(
        map(
            lambda result: {
                "symbol": result['T'],
                "price": result['c']
            },
            results
        ),
        reverse=True,
        key=sort_func
    )[:10]

def get_top_10_trading_volumes(best=True):
    def sort_func(e):
        return e['volume']
    

    return sorted(
        map(
            lambda result: {
                "symbol": result['T'],
                "volume": result['v']
            },
            results
        ),
        reverse=best,
        key=sort_func
    )[:10]




if __name__ == "__main__":

    results = extract_results('input.json')
    top_10_closing_prices = get_top_10_closing_prices(results=results)

    categories = [closing_price["symbol"] for closing_price in top_10_closing_prices]
    values = [closing_price["price"] for closing_price in top_10_closing_prices]

    top_10_best_trading_volumes = get_top_10_trading_volumes(best=True)
    top_10_worst_trading_volumes = get_top_10_trading_volumes(best=False)
    table_top_10_best_trading_volumes_data = [['Symbol', 'Volume']]
    table_top_10_worst_trading_volumes_data = [['Symbol', 'Volume']]

    for best_trading_volume in top_10_best_trading_volumes:
        table_top_10_best_trading_volumes_data.append([best_trading_volume["symbol"], best_trading_volume["volume"]])
    
    for worst_trading_volume in top_10_worst_trading_volumes:
        table_top_10_worst_trading_volumes_data.append([worst_trading_volume["symbol"], worst_trading_volume["volume"]])

    plt.figure(figsize=(8, 4))
    plt.bar(categories, values)
    
    plt.title('Top 10 Closing Prices')
    plt.xlabel('Stock')
    plt.ylabel('Values')

    # Save the bar chart as an image
    plt.savefig('bar_chart.png')

    # Create a PDF document
    pdf_filename = 'report.pdf'
    c = canvas.Canvas(pdf_filename, pagesize=letter)
    c.setFillColor(colors.grey)
    c.setFont("Helvetica-Bold", 24)
    c.drawString(230, 760, "Stock Report")

    image_path = os.path.join(os.getcwd(), "bar_chart.png")
    c.drawImage(image_path, 50, 400, width=500, height=265)

    c.setFont("Helvetica-Bold", 14)

    c.drawString(50, 270, "Top 10 Best Trading Volumes")
    table_top_10_best_trading_volumes = Table(table_top_10_best_trading_volumes_data)
    table_top_10_best_trading_volumes.wrapOn(c, 200, 350)
    table_top_10_best_trading_volumes.drawOn(c, 150, 20)

    
    c.showPage()
    c.setFillColor(colors.grey)
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 760, "Top 10 Worst Trading Volumes")
    table_top_10_worst_trading_volumes = Table(table_top_10_worst_trading_volumes_data)
    table_top_10_worst_trading_volumes.wrapOn(c, 200, 350)
    table_top_10_worst_trading_volumes.drawOn(c, 170, 475)

    c.save()
    print("PDF report generated successfully.")

    os.remove('bar_chart.png')
    os.rename("report.pdf", "/data/outputs/report.pdf")
    print("PDF report moved to /data/outputs.")

