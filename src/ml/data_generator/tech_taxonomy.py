#src/ml/data_generator/tech_taxonomy.py
"""
Tech skills taxonomy and job/resume templates.
Handles all the domain knowledge about tech skills, job types, and resume structures.
"""

class TechSkillsTaxonomy:
    """Comprehensive tech skills taxonomy."""
    def __init__(self): 
        self._initialize_skills()
        self._initialize_job_templates()
        self._initialize_resume_templates()
    
    def _initialize_skills(self): 
        """Initialize comprehensive tech skills taxonomy."""
        self.programming_languages = {
            'backend': ['Python', 'Java', 'Go', 'Rust', 'C++', 'C#', 'Scala', 'Ruby', 'PHP'],
            'frontend': ['JavaScript', 'TypeScript', 'HTML5', 'CSS3', 'Dart'],
            'mobile': ['Swift', 'Kotlin', 'Flutter', 'React Native', 'Objective-C'],
            'data': ['Python', 'R', 'SQL', 'Scala', 'Julia', 'MATLAB'],
            'systems': ['C', 'C++', 'Rust', 'Go', 'Assembly', 'Bash']
        }
        self.frameworks = {
            'web_backend': ['Django', 'Flask', 'FastAPI', 'Spring Boot', 'Express.js', 'Ruby on Rails', 'ASP.NET'],
            'web_frontend': ['React', 'Angular', 'Vue.js', 'Svelte', 'Next.js', 'Nuxt.js', 'Ember.js'],
            'mobile': ['React Native', 'Flutter', 'Xamarin', 'Ionic', 'NativeScript'],
            'ml_ai': ['PyTorch', 'TensorFlow', 'Scikit-learn', 'Keras', 'Hugging Face', 'OpenCV', 'spaCy'],
            'data': ['Pandas', 'NumPy', 'Apache Spark', 'Kafka', 'Airflow', 'Dask']
        }
        self.databases = {
            'relational': ['PostgreSQL', 'MySQL', 'SQLite', 'Oracle', 'SQL Server'],
            'nosql': ['MongoDB', 'Redis', 'Elasticsearch', 'Cassandra', 'DynamoDB', 'Neo4j'],
            'data_warehouses': ['Snowflake', 'BigQuery', 'Redshift', 'ClickHouse']
        }
        self.cloud_devops = {
            'cloud_platforms': ['AWS', 'Google Cloud Platform', 'Microsoft Azure', 'DigitalOcean'],
            'containerization': ['Docker', 'Kubernetes', 'Podman', 'OpenShift'],
            'ci_cd': ['Jenkins', 'GitHub Actions', 'GitLab CI', 'CircleCI', 'Travis CI'],
            'infrastructure': ['Terraform', 'Ansible', 'Puppet', 'Chef', 'CloudFormation'],
            'monitoring': ['Prometheus', 'Grafana', 'Datadog', 'New Relic', 'ELK Stack']
        }
        self.tools = {
            'version_control': ['Git', 'GitHub', 'GitLab', 'Bitbucket', 'SVN'],
            'testing': ['Jest', 'PyTest', 'JUnit', 'Selenium', 'Cypress', 'Postman'],
            'design': ['Figma', 'Sketch', 'Adobe XD', 'InVision'],
            'project_management': ['Jira', 'Confluence', 'Trello', 'Asana', 'Linear']
        }
        self.certifications = {
            'cloud': ['AWS Solutions Architect', 'AWS Developer', 'GCP Professional Developer', 'Azure Developer'],
            'security': ['CISSP', 'CISM', 'CompTIA Security+', 'CEH'],
            'project_management': ['PMP', 'Scrum Master', 'Product Owner'],
            'data': ['Google Data Analytics', 'Tableau Desktop Specialist', 'Microsoft Data Analyst']
        }

    def _initialize_job_templates(self): 
        """Initialize job description templates for different tech roles."""
        self.job_templates = {
            'software_engineer': {
                'titles': ['Software Engineer', 'Software Developer', 'Full Stack Developer', 'Backend Developer'],
                'core_skills': ['programming_languages.backend', 'frameworks.web_backend', 'databases.relational', 'tools.version_control'],
                'preferred_skills': ['cloud_devops.cloud_platforms', 'tools.testing', 'cloud_devops.containerization'],
                'responsibilities': [
                    'Design and develop scalable web applications',
                    'Write clean, maintainable, and well-documented code',
                    'Collaborate with cross-functional teams on product features',
                    'Participate in code reviews and architectural discussions',
                    'Debug and resolve software issues in production environments'
                ]
            },
            'frontend_engineer': {
                'titles': ['Frontend Engineer', 'Frontend Developer', 'UI Developer', 'React Developer'],
                'core_skills': ['programming_languages.frontend', 'frameworks.web_frontend', 'tools.version_control'],
                'preferred_skills': ['tools.design', 'tools.testing', 'frameworks.mobile'],
                'responsibilities': [
                    'Build responsive and interactive user interfaces',
                    'Collaborate with designers to implement pixel-perfect designs',
                    'Optimize applications for maximum speed and scalability',
                    'Ensure cross-browser compatibility and accessibility standards',
                    'Integrate with backend APIs and services'
                ]
            },
            'data_scientist': {
                'titles': ['Data Scientist', 'ML Engineer', 'Machine Learning Engineer', 'AI Engineer'],
                'core_skills': ['programming_languages.data', 'frameworks.ml_ai', 'frameworks.data', 'databases.nosql'],
                'preferred_skills': ['cloud_devops.cloud_platforms', 'databases.data_warehouses', 'tools.version_control'],
                'responsibilities': [
                    'Develop machine learning models for business problems',
                    'Analyze large datasets to extract actionable insights',
                    'Build data pipelines for model training and inference',
                    'Collaborate with engineering teams to deploy ML models',
                    'Present findings and recommendations to stakeholders'
                ]
            },
            'devops_engineer': {
                'titles': ['DevOps Engineer', 'Site Reliability Engineer', 'Platform Engineer', 'Cloud Engineer'],
                'core_skills': ['cloud_devops.cloud_platforms', 'cloud_devops.containerization', 'cloud_devops.infrastructure'],
                'preferred_skills': ['cloud_devops.ci_cd', 'cloud_devops.monitoring', 'programming_languages.systems'],
                'responsibilities': [
                    'Design and maintain CI/CD pipelines',
                    'Manage cloud infrastructure and deployments',
                    'Monitor system performance and reliability',
                    'Automate operational processes and workflows',
                    'Ensure security best practices across infrastructure'
                ]
            },
            'mobile_developer': {
                'titles': ['Mobile Developer', 'iOS Developer', 'Android Developer', 'Mobile Engineer'],
                'core_skills': ['programming_languages.mobile', 'frameworks.mobile', 'tools.version_control'],
                'preferred_skills': ['tools.design', 'cloud_devops.cloud_platforms', 'tools.testing'],
                'responsibilities': [
                    'Develop native and cross-platform mobile applications',
                    'Integrate with backend services and APIs',
                    'Optimize app performance and user experience',
                    'Publish apps to app stores and manage releases',
                    'Debug and fix issues reported by users'
                ]
            }
        }

    def _initialize_resume_templates(self): 
        """Initialize resume templates for different experience levels."""
        self.resume_templates = {
            'entry': {
                'experience_years_range': (0, 2),
                'num_jobs_range': (0, 2),
                'num_projects_range': (2, 4),
                'education_level': ['Bachelor', 'Master'],
                'summary_templates': [
                    "Recent graduate with a passion for {primary_skill} development and {secondary_skill}",
                    "Entry-level {job_category} with strong foundation in {primary_skill} and {secondary_skill}",
                    "Computer science graduate seeking to leverage {primary_skill} and {secondary_skill} skills"
                ]
            },
            'mid': {
                'experience_years_range': (2, 5),
                'num_jobs_range': (2, 3),
                'num_projects_range': (3, 5),
                'education_level': ['Bachelor', 'Master'],
                'summary_templates': [
                    "Experienced {job_category} with {years} years developing {primary_skill} applications",
                    "Mid-level developer specializing in {primary_skill} and {secondary_skill} technologies",
                    "{years}+ years of experience in {primary_skill} development and system design"
                ]
            },
            'senior': {
                'experience_years_range': (5, 10),
                'num_jobs_range': (3, 4),
                'num_projects_range': (4, 6),
                'education_level': ['Bachelor', 'Master', 'PhD'],
                'summary_templates': [
                    "Senior {job_category} with {years} years of experience leading {primary_skill} development",
                    "Technical lead with expertise in {primary_skill}, {secondary_skill}, and team mentoring",
                    "Seasoned professional with {years} years building scalable {primary_skill} systems"
                ]
            },
            'staff': {
                'experience_years_range': (8, 15),
                'num_jobs_range': (4, 5),
                'num_projects_range': (5, 8),
                'education_level': ['Bachelor', 'Master', 'PhD'],
                'summary_templates': [
                    "Staff engineer with {years} years architecting large-scale {primary_skill} systems",
                    "Technical leader driving {primary_skill} and {secondary_skill} initiatives across teams",
                    "Principal engineer with deep expertise in {primary_skill} and system architecture"
                ]
            }
        }