from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import col, count, min

# Create a Spark session
spark = SparkSession.builder.appName("Assignment 1").getOrCreate()

# Task 1
# Part of the code for changing the time and date works, but saving the changes does not.
'''
file_name = "dataset.txt"
df = spark.read.option("header", "true").csv(file_name)
df = df.withColumn(
    "Timestamp",
    col("Timestamp").cast("double") + 28800
).withColumn(
    "NewTime",
    expr(
        "from_unixtime(unix_timestamp(Time, 'HH:mm:ss') + 8 * 3600, 'HH:mm:ss')"
    )
).withColumn(
    "OldTime",
    expr(
        "from_unixtime(unix_timestamp(Time, 'HH:mm:ss'), 'HH:mm:ss')"
    )
).withColumn(
    "Date",
    when(
        col("NewTime") < col("OldTime"),
        expr("date_add(Date, 1)"),
    ).otherwise(col("Date"))
).drop("OldTime")

df.coalesce(1).write.mode("overwrite").csv(file_name, header=True, sep=",")
'''

# Task 2
file_path = "dataset.txt"
data_df = spark.read.csv(file_path, header=True, inferSchema=True)
count_df = data_df.groupBy("UserID", "Date").count()
window_spec = Window.partitionBy("UserID").orderBy("Date")
cumulative_count_df = count_df.withColumn("cumulative_count", F.sum("count").over(window_spec))
filtered_df = cumulative_count_df.filter("cumulative_count >= 5")
result_df = filtered_df.groupBy("UserID").agg(F.countDistinct("Date").alias("num_days"))
result_df = result_df.orderBy(F.desc("num_days"), "UserID").limit(5)

# Show the result for Task 2
result_df.show()

# Task 3
file_path = 'dataset.txt'
df = spark.read.csv(file_path, header=True)
df = df.withColumn("Date", col("Date").cast("date"))
result_df = df.groupBy("UserID", col("Date").alias("Week")).agg(count("*").alias("DataPoints"))
result_df = result_df.filter(result_df.DataPoints > 100)
final_result_df = result_df.groupBy("UserID").agg(count("*").alias("WeeksWithMoreThan100DataPoints"))

# Show the result for Task 3
final_result_df.show()

# Task 4
file_path = "dataset.txt"
df = spark.read.csv(file_path, header=True, inferSchema=True)
df = df.withColumn("Latitude", col("Latitude").cast("double"))
southernmost_points = df.groupBy("UserID").agg(min(col("Latitude")).alias("SouthernmostLatitude"), min(col("Date")).alias("AchievedDate"))
result = southernmost_points.orderBy("SouthernmostLatitude", "AchievedDate").limit(5)

# Show the result for Task 4
result.show()

# Task 5
file_path = "dataset.txt"
df = spark.read.csv(file_path, header=True, inferSchema=True)
df = df.withColumn("DateTime", F.concat(F.col("Date"), F.lit(" "), F.col("Time")))
df = df.withColumn("Timestamp", F.to_timestamp("DateTime", "yyyy-MM-dd HH:mm:ss"))
window_spec = Window.partitionBy("UserID").orderBy("Timestamp")
df = df.withColumn("AltitudeSpan", F.max("Altitude").over(window_spec))
result_df = df.groupBy("UserID").agg(F.max("AltitudeSpan").alias("MaxAltitudeSpan"))
top5_users = result_df.orderBy(F.desc("MaxAltitudeSpan")).limit(5)

# Show the result for Task 5
top5_users.show()

# Task 6
file_path = "dataset.txt"
df = spark.read.csv(file_path, header=True, inferSchema=True)
windowSpec = Window().partitionBy("UserID").orderBy("Timestamp")
df = df.withColumn("prev_lat", F.lag("Latitude").over(windowSpec))
df = df.withColumn("prev_lon", F.lag("Longitude").over(windowSpec))
R = 6371
lat_diff = F.radians(df["Latitude"] - df["prev_lat"])
lon_diff = F.radians(df["Longitude"] - df["prev_lon"])
a = F.sin(lat_diff / 2) ** 2 + F.cos(F.radians(df["prev_lat"])) * F.cos(F.radians(df["Latitude"])) * F.sin(lon_diff / 2) ** 2
c = 2 * F.atan2(F.sqrt(a), F.sqrt(1 - a))
distance = R * c
df = df.withColumn("Distance", distance)
total_distance = df.select(F.sum("Distance")).collect()[0][0]
print(f"Total distance traveled by all users on all days: {total_distance} kilometers")
max_distance_day = df.groupBy("UserID", "Date").agg(F.sum("Distance").alias("TotalDistance")). \
    withColumn("max_distance", F.max("TotalDistance").over(Window().partitionBy("UserID"))). \
    filter(F.col("TotalDistance") == F.col("max_distance")). \
    select("UserID", "Date", "TotalDistance")

# Show the result for Task 6
max_distance_day.show()

# Task 7
file_path = "dataset.txt"
df = spark.read.csv(file_path, header=True, inferSchema=True)
df = df.withColumn("Timestamp", F.to_timestamp(df["Timestamp"]))
window_spec_speed = Window.partitionBy("UserID").orderBy(F.desc("speed"))
df = df.withColumn("lag_time", F.lag("Timestamp").over(window_spec_speed))
df = df.withColumn("time_diff", (F.col("Timestamp").cast("long") - F.col("lag_time").cast("long")) / 3600)
df = df.withColumn("distance", F.sqrt((F.col("Latitude") - F.lag("Latitude").over(window_spec_speed))**2 +
                                      (F.col("Longitude") - F.lag("Longitude").over(window_spec_speed))**2))
df = df.withColumn("speed", F.when((F.col("time_diff") > 0) & (F.col("distance") > 0),
                                   F.col("distance") / F.col("time_diff")).otherwise(None))
max_speed_window = Window.partitionBy("UserID").orderBy(F.desc("speed"))
result_df_speed = df.withColumn("rank", F.row_number().over(max_speed_window)) \
    .filter("rank = 1") \
    .select("UserID", "speed", "Date", "Time")

# Show the result for Task 7
result_df_speed.show(truncate=False)

# Stop the Spark session
spark.stop()
