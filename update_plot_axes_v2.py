import os

file_path = 'ui/analysis/view.py'

with open(file_path, 'r') as f:
    content = f.read()

# Helper for standard axis style
# We use double quotes for internal strings to match potential python repr or just consistency
# But content replacement must match EXACTLY what is in the file.

# 1. 1D Histogram (fig_g)
# Previous update:
# xaxis=dict(showgrid=True, gridcolor='lightgrey', color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
# yaxis=dict(showgrid=False, showticklabels=False, color='black', showline=True, linewidth=1, linecolor='black', mirror=True)

search_1 = "xaxis=dict(showgrid=True, gridcolor='lightgrey', color='black', showline=True, linewidth=1, linecolor='black', mirror=True),\n                                yaxis=dict(showgrid=False, showticklabels=False, color='black', showline=True, linewidth=1, linecolor='black', mirror=True)"

replace_1 = "xaxis=dict(showgrid=True, gridcolor='lightgrey', color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showline=True, linewidth=1, linecolor='black', mirror=True),\n                                yaxis=dict(showgrid=False, showticklabels=False, color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showline=True, linewidth=1, linecolor='black', mirror=True)"

# 2. Interactive KDE
# Previous update:
# xaxis=dict(fixedrange=True, color='black', showline=True, linewidth=1, linecolor='black', mirror=True),
# yaxis=dict(fixedrange=True, color='black', showline=True, linewidth=1, linecolor='black', mirror=True),

search_2 = "xaxis=dict(fixedrange=True, color='black', showline=True, linewidth=1, linecolor='black', mirror=True),\n                            yaxis=dict(fixedrange=True, color='black', showline=True, linewidth=1, linecolor='black', mirror=True),"

replace_2 = "xaxis=dict(fixedrange=True, color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=True, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='black', mirror=True),\n                            yaxis=dict(fixedrange=True, color='black', tickfont=dict(color='black'), title_font=dict(color='black'), showgrid=True, gridcolor='lightgrey', showline=True, linewidth=1, linecolor='black', mirror=True),"

# 3. 2D Quadrant
# Previous update:
# xaxis=dict(color="black", showline=True, linewidth=1, linecolor="black", mirror=True),
# yaxis=dict(color="black", showline=True, linewidth=1, linecolor="black", mirror=True)

search_3 = 'xaxis=dict(color="black", showline=True, linewidth=1, linecolor="black", mirror=True),\n                        yaxis=dict(color="black", showline=True, linewidth=1, linecolor="black", mirror=True)'

replace_3 = 'xaxis=dict(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), showgrid=True, gridcolor="lightgrey", showline=True, linewidth=1, linecolor="black", mirror=True),\n                        yaxis=dict(color="black", tickfont=dict(color="black"), title_font=dict(color="black"), showgrid=True, gridcolor="lightgrey", showline=True, linewidth=1, linecolor="black", mirror=True)'

# 4. 2D Polygon
# Same string as 3?
# In my previous script I replaced them identically.
# Let's check the file content first to be sure they are identical.
# Actually search_3 appears TWICE in the file if I replaced both 2D Quadrant and 2D Polygon with the same string in previous step.
# So simple replace of search_3 should cover both if they are identical strings.

new_content = content.replace(search_1, replace_1)
new_content = new_content.replace(search_2, replace_2)
new_content = new_content.replace(search_3, replace_3)

if new_content != content:
    with open(file_path, 'w') as f:
        f.write(new_content)
    print("Successfully updated plot axes with explicit black labels and lightgrey grids.")
else:
    print("No changes made. Please check search strings.")
    if search_1 not in content: print("Search 1 failed")
    if search_2 not in content: print("Search 2 failed")
    if search_3 not in content: print("Search 3 failed")

