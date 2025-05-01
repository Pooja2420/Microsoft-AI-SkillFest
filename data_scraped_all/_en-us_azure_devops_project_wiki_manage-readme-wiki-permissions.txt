Note
Access to this page requires authorization. You can trysigning inorchanging directories.
Access to this page requires authorization. You can trychanging directories.
Manage wiki permissions
Article
2024-09-06
7 contributors
In this article
Azure DevOps Services | Azure DevOps Server 2022 - Azure DevOps Server 2019
In this article, learn about managing permissions for your wiki. By default, all members of the Contributors group can edit wiki pages.
Manage permissions for wikis
By default, all project contributors have "read" and "edit" access to the wiki repository. You can manage these permissions to control who can read and edit wiki pages. For more information, seeGet started with permissions, access, and security groups.
Sign in to your project (https://dev.azure.com/{Your_Organization/Your_Project}).
https://dev.azure.com/{Your_Organization/Your_Project}
SelectWiki>More actions>Wiki security.

For definitions of each repository permission, seeGit repository permissions.

If you don't have access to create a wiki page, contact an Administrator, who can grant you adequate permission on the underlying Git repository of the wiki.
Grant Edit permissions to an individual or group
To grantEditpermissions to an individual or group, do the following steps.
Sign in to your project (https://dev.azure.com/{Your_Organization/Your_Project}).
https://dev.azure.com/{Your_Organization/Your_Project}
SelectWiki>More actions>Wiki security.

SelectAdd. If this button isn't available, check yourpermissions.
Enter the name of the user or group you want to grant permissions to and select the user or group from the search results.
After you add the user or group, they're listed in the Wiki security pane.
To grantEditpermissions, set theContribute permissiontoAllow.
Savethe changes.
Other considerations
Ensure that the user or group has the necessary access level to the Azure DevOps project.
Review and adjust other permissions as needed to maintain security and proper access control, such asRead,Delete, andManage.
Stakeholder wiki access
Private projects
Users withStakeholder accessin a private project can readprovisionedwiki pages and view revisions, but they can't edit them. For example, Stakeholders can't create, edit, reorder, or revert changes to project wiki pages. These permissions can't be changed.
Stakeholders have no access to read or editpublished codewiki pages in private projects. For more information, see theStakeholder access quick reference for project and code wikis.
Public projects
Stakeholders have full access to wikis in public projects.
For more information about Stakeholder access, seeAbout access levels, Stakeholder access, Public versus private feature access.
Note
You can set permissions for the entire wiki, but not for individual pages.
Related articles
Learn default Git repository and branch permissions
Get Started with Git
Learn about Azure DevOps security
Feedback
Was this page helpful?
Additional resources