filename = 'linksFile_v2_fixed.dat'

file = open(filename, 'r')
new_file = open('linksFile_v2_fixed_speed.dat', 'w')

speed_limits = [
    (0, 3000, 11.11),  # 0-3 km: 40 km/h
    (3000, 7000, 19.44),  # 3-7 km: 70 km/h
    (7000, 10600, 11.11),  # 7-10.6 km: 40 km/h
    (10600, 20400, 16.67),  # 10.6-20.4 km: 60 km/h
    (20400, 24100, 22.22),  # 20.4-24.1 km: 80 km/h
    (24100, 33300, 22.22),  # 24.1-30 km: 80 km/h
    (33300, 39100, 16.67),  # 30-40 km: 60 km/h
    (39100, 43300, 22.22),  # 40-50 km: 80 km/h
    (43300, 50500, 22.22),  # 50-60 km: 80 km/h
    (50500, 76300, 11.11),  # 60-70 km: 40 km/h
]

current_distance = 0
for (i, line) in enumerate(file):
    if i < 2:  # skip header lines
        new_file.write(line)
        continue
    current_distance += 50
    speed_limit = 0
    for (start, end, speed) in speed_limits:
        if start <= current_distance < end:
            speed_limit = speed
            break
    new_lines = line.replace('19.4', f'{speed_limit:.2f}')
    new_file.write(new_lines)
file.close()
new_file.close()
