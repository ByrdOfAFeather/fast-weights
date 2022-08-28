class Solution:
	def romanToInt(self, s: str) -> int:
		numeral_to_int = {
			"I": 1,
			"V": 5,
			"X": 10,
			"L": 50,
			"C": 100,
			"D": 500,
			"M": 1000
		}
		special_cases = {"I", "X", "C"}
		running_sum = 0
		check_next = False
		i_case, x_case, c_case = False, False, False
		for idx, char in enumerate(s):
			if char in special_cases and idx < len(s) - 1 and not (x_case or i_case or c_case):
				next_val = s[idx + 1]
				i_case = char == "I" and (next_val == "V" or next_val == "X")
				x_case = char == "X" and (next_val == "L" or next_val == "C")
				c_case = char == "C" and (next_val == "D" or next_val == "M")
				if i_case or x_case or c_case:
					continue

			cur_value = numeral_to_int[char]
			if i_case or x_case or c_case:
				if i_case:
					running_sum += (cur_value - 1)
				if x_case:
					print("HERE")
					print(cur_value - 10)
					running_sum += (cur_value - 10)
					print(running_sum)
				if c_case:

					running_sum += (cur_value - 100)
				else:
					running_sum += cur_value + numeral_to_int[s[idx - 1]]
				i_case, x_case, c_case = False, False, False
			else:
				running_sum += cur_value
		return running_sum


Solution().romanToInt("MCMXCIV")