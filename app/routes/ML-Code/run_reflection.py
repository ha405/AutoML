
from run_reflect_cycle import run_reflection_cycle


csv_file_path = 'CarPrice_Assignment.csv'  # path to our csv file
problem_description = "We are required to model the price of cars with the available independent variables. It will be used by the management to understand how exactly the prices vary with the independent variables. They can accordingly manipulate the design of the cars, the business strategy etc. to meet certain price levels. Further, the model will be a good way for management to understand the pricing dynamics of a new market."
#Write the problem statement as a string above 

# Run the reflection cycle
final_refined_code = run_reflection_cycle(csv_file_path, problem_description, max_reflections=3)

# Print the final refined ML code
print("Final Refined ML Code:\n", final_refined_code)

# Save the final refined ML code to a .py file
with open("Final_ML_code.py", "w", encoding="utf-8") as f:
    f.write(final_refined_code)

print("Final refined ML code has been saved to Final_ML_code.py")
