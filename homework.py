def calculate_total(price, tax):
    return price + tax

def process_order(order_id, price):
    # BUG: Hardcoded tax is too high!
    tax = 999 
    total = calculate_total(price, tax)
    save_to_db(total)

def save_to_db(amount):
    print("Saving to database: " + str(amount))