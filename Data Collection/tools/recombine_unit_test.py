# recombine_unit_test.py - Logan Reichling - Start 10/5/25 - UC DaSec
# Implements recombine_unit.v in python for testing

def recombine_unit(hex_input, d=2, count=128):
    """
    Test recombine_unit module in Python. Converts bit matrix share format back to normal
    :param hex_input: Hex string input with bits that should correspond to the count parameter
    :param d: Number of total shares
    :param count: Number of total output bits in output hex number
    :return: <count> bit hex number as string (<count>//4 hex characters)
    """
    input_num = int(hex_input, 16)
    binary_str = bin(input_num)[2:].zfill(d*count)

    result_bits = list()
    for i in range(count):
        start_pos = i * d
        group_bits = binary_str[start_pos:start_pos + d]

        xor_result = 0
        for bit in group_bits:
            xor_result ^= int(bit)
        result_bits.append(str(xor_result))
    result_binary = ''.join(result_bits)
    result_hex = hex(int(result_binary, 2))[2:].zfill(count//4)
    return result_hex


# Main
if __name__ == "__main__":
    result = recombine_unit("b12815f08451fabe9a4fa5da5960024e88ee8c62388dd94fb5003476e7341e2b", d=2, count=128)
    print(f"Custom: {result}")  # LSB