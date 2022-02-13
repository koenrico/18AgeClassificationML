# Author: Rico Koen
# Description: This script is used to prepare a data set for Machine Learning, it will split the full data set into a training set and a testing set

#!/usr/bin/perl -w
use strict;
use warnings;
use utf8;
use File::Find;
use Data::Dumper;
use POSIX;
use File::Copy qw(copy);
use File::Path qw(make_path remove_tree);

# Denine common variables
my @content;
my %all_data;
# Set the categories for under and over 18
my %categories = ( 
        under => { age_start => 0, age_end => 17 }, 
        over => { age_start => 18, age_end => 999 }
    );

# Set the dataset path
my $data_path = "/home/ctext/Desktop/rico/AgeEstimation/";

# Get all directories and files in the dataset
sub dir_list {
  push @content, $File::Find::name;
  return;
}

eval
{
    find( \&dir_list, $data_path . "images_cropped");

    # Loop through the files and folders
    foreach ( @content ) {
        # Get the filename or foldername from the path
        my @parts = split('/', $_);
        my $length = scalar @parts;
        # Make sure we only work with files
        if (index($parts[$length-1], '.') != -1) {
            my @fileparts = split('_', $parts[$length-1]);
            # Get the age from the filename
            my $age = $fileparts[0];
            # Initialise the category varibale
            my $category = undef;
            # Make sure the age is a number and get the category the file falls into
            if ($age =~ /^?\d+$/) {
                my $int_age = sprintf("%d", $age);
                for my $age_category (keys %categories) { 
                    if ($int_age >= $categories{$age_category}{'age_start'} && $int_age <= $categories{$age_category}{'age_end'}) {
                        $category = $age_category;
                    }
                }
            }
            # Add the category to a hash and set the value to the filename
            push( @{ $all_data { $category } }, $_);
        }
    }

    # Loop through the categories
    foreach my $category (sort(keys %all_data))
    {
        # Get the total images in for the category
        my @images = @{$all_data{$category}};
        my $total_images = scalar @images;
        my $total_copied = 0;

        # Copy 10% of the images to the testing folder
        for (my $i = 0; $i < ceil($total_images*0.1); $i++) {
            my @parts = split('/', $images[$i]);
            my $length = scalar @parts;
            my $destination_path = $data_path . "utkface_18_preprocessed/testing/" . $category;
            my $destination_file = $data_path . "utkface_18_preprocessed/testing/" . $category . "/" . lc($parts[$length-1]);
            make_path($destination_path, {
                chmod => 0777,
            });
            copy $images[$i], $destination_file;
            $total_copied++;
        }
        my $total_copied_testing = $total_copied;
        # Copy 20% of the images to the testing folder
        for (my $i = $total_copied; $i < (ceil($total_images*0.2) + $total_copied_testing); $i++) {
            my @parts = split('/', $images[$i]);
            my $length = scalar @parts;
            my $destination_path = $data_path . "utkface_18_preprocessed/validation/" . $category;
            my $destination_file = $data_path . "utkface_18_preprocessed/validation/" . $category . "/" . lc($parts[$length-1]);
            make_path($destination_path, {
                chmod => 0777,
            });
            copy $images[$i], $destination_file;
            $total_copied++;
        }
        # Copy 70% of the images to the training folder
        for (my $i = $total_copied; $i < $total_images; $i++) {
            my @parts = split('/', $images[$i]);
            my $length = scalar @parts;
            my $destination_path = $data_path . "utkface_18_preprocessed/training/" . $category;
            my $destination_file = $data_path . "utkface_18_preprocessed/training/" . $category . "/" . lc($parts[$length-1]);
            make_path($destination_path, {
                chmod => 0777,
            });
            copy $images[$i], $destination_file;
            $total_copied++;
        }
    }
}