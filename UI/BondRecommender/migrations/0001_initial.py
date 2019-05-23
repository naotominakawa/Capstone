# Generated by Django 2.1.7 on 2019-02-19 06:52

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Currency',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('symbol', models.CharField(max_length=3)),
            ],
        ),
        migrations.CreateModel(
            name='Rates',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('x_currency', models.CharField(max_length=3)),
                ('rate', models.FloatField(default=1.0)),
                ('last_update_time', models.DateTimeField()),
                ('currency', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='BondRecommender.Currency')),
            ],
        ),
        migrations.CreateModel(
            name='Securities',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('isin', models.CharField(max_length=16)),
                ('YAS_price', models.FloatField()),
                ('OAS_spread', models.FloatField()),
                ('modified_duration', models.FloatField()),
                ('G_spread', models.FloatField()),
                ('yld', models.FloatField()),
            ],
        ),
    ]
