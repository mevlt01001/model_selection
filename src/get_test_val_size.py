from src.log import log

def test_val_size():
    while True:    
        try:
            test_size = float(input("Please enter the test size (e.g., 0.2): "))
            log("FUNC_TEST_VAL_SIZE", f"Test size: {test_size}")
            val_size = float(input("Please enter the validation size (e.g., 0.2): "))
            log("FUNC_TEST_VAL_SIZE", f"Validation size: {val_size}")

            if test_size + val_size > 1:
                raise ValueError("The sum of test and validation sizes cannot exceed 1.")
            else:
                return test_size, val_size
        except Exception as e:
            print(f"Hata meydana geldi: {e}")
            log("FUNC_TEST_VAL_SIZE", f"Hata meydana geldi: {e}")
            continue

